# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId, JobResult

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_task_cpu_s",
        "_task_wall_s",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._task_cpu_s = 0.0
        self._task_wall_s = 0.0
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, success: bool, task_cpu_s: float, task_wall_s: float
    ) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            self._task_cpu_s += max(task_cpu_s, 0.0)
            self._task_wall_s += max(task_wall_s, 0.0)
            return self._completed == self._n_tasks, self._success

    def result(self) -> JobResult:
        return JobResult(
            job_id=self._job_id,
            success=self._success,
            task_cpu_s=self._task_cpu_s,
            task_wall_s=self._task_wall_s,
            task_count=self._completed,
        )


def _set_native_thread_name(name: str) -> None:
    """Expose the worker's name as the kernel comm (15 bytes) so per-thread
    CPU accounting from /proc can attribute I/O worker time."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(15, name.encode("utf-8")[:15], 0, 0, 0)
    except Exception:
        return


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._active_tasks = 0
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[JobResult] = deque()

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._load_q.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[JobResult]:
        with self._condition:
            jobs = list(self._finished_q)
            self._finished_q.clear()
            return jobs

    def wait_idle(self) -> None:
        """Wait for all queued and currently executing tasks to finish."""
        with self._condition:
            self._condition.wait_for(
                lambda: not self._load_q
                and not self._store_q
                and self._active_tasks == 0
            )

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        _set_native_thread_name(threading.current_thread().name)
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                task, state = primary.popleft() if primary else secondary.popleft()
                self._active_tasks += 1
            wall_start = time.perf_counter()
            cpu_start = time.thread_time()
            success = True
            try:
                task()
            except Exception as exc:
                success = False
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
            finally:
                task_cpu_s = time.thread_time() - cpu_start
                task_wall_s = time.perf_counter() - wall_start
                job_finished, _ = state.task_done(
                    success, task_cpu_s, task_wall_s
                )

            with self._condition:
                if job_finished:
                    self._finished_q.append(state.result())
                self._active_tasks -= 1
                if (
                    not self._load_q
                    and not self._store_q
                    and self._active_tasks == 0
                ):
                    self._condition.notify_all()
