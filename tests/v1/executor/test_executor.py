# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import fcntl
import multiprocessing
import os
import pickle
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.external_load_completion import (
    ExternalLoadCompletionNotice,
)
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    WorkerProc,
    _resize_external_load_completion_pipe,
)
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


def _spawn_send_external_load_completion_notice(writer, notice) -> None:
    writer.send_bytes(pickle.dumps(notice, protocol=pickle.HIGHEST_PROTOCOL))
    writer.close()


def test_external_load_completion_notice_spawn_pipe_round_trip():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    notice = ExternalLoadCompletionNotice(11, 2, 19, "request-19", 1.25)
    process = context.Process(
        target=_spawn_send_external_load_completion_notice,
        args=(writer, notice),
    )
    try:
        process.start()
        writer.close()
        assert reader.poll(30)
        assert pickle.loads(reader.recv_bytes(4092)) == notice
        process.join(30)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        reader.close()
        process.close()


def test_external_load_completion_pipe_is_enlarged_best_effort():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    try:
        before = fcntl.fcntl(reader.fileno(), fcntl.F_GETPIPE_SZ)
        after = _resize_external_load_completion_pipe(reader)
        assert after >= before
        assert after == fcntl.fcntl(reader.fileno(), fcntl.F_GETPIPE_SZ)
    finally:
        writer.close()
        reader.close()


def test_external_load_completion_pipe_send_is_fail_open():
    worker = object.__new__(WorkerProc)
    worker.rank = 0
    worker._external_load_completion_epoch = 7
    worker._external_load_completion_writer = MagicMock()
    worker._external_load_completion_max_payload_bytes = 4092
    worker._external_load_completion_writer.send_bytes.side_effect = (
        BlockingIOError
    )

    assert not worker._emit_external_load_completion_notice(3, "request-3")


def test_external_load_completion_oversized_notice_does_not_touch_pipe():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    os.set_blocking(writer.fileno(), False)
    worker = object.__new__(WorkerProc)
    worker.rank = 0
    worker._external_load_completion_epoch = 7
    worker._external_load_completion_writer = writer
    worker._external_load_completion_max_payload_bytes = 4092
    try:
        assert not worker._emit_external_load_completion_notice(
            3, "x" * 8192
        )
        assert not reader.poll()
        assert worker._emit_external_load_completion_notice(4, "request-4")
        assert reader.poll(5)
        notice = pickle.loads(reader.recv_bytes(4092))
        assert notice.job_id == 4
        assert notice.req_id == "request-4"
    finally:
        writer.close()
        reader.close()


def test_external_load_completion_full_pipe_fails_without_partial_frame():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    os.set_blocking(writer.fileno(), False)
    worker = object.__new__(WorkerProc)
    worker.rank = 0
    worker._external_load_completion_epoch = 7
    worker._external_load_completion_writer = writer
    worker._external_load_completion_max_payload_bytes = 4092
    sent = 0
    try:
        for job_id in range(10_000):
            if not worker._emit_external_load_completion_notice(
                job_id, f"request-{job_id}"
            ):
                break
            sent += 1
        assert 0 < sent < 10_000

        received = []
        while reader.poll():
            received.append(pickle.loads(reader.recv_bytes(4092)))
        assert len(received) == sent
        assert [notice.job_id for notice in received] == list(range(sent))

        assert worker._emit_external_load_completion_notice(
            10_001, "request-after-drain"
        )
        assert reader.poll(5)
        assert pickle.loads(reader.recv_bytes(4092)).job_id == 10_001
    finally:
        writer.close()
        reader.close()


def test_external_load_completion_concurrent_writers_publish_atomically():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    _resize_external_load_completion_pipe(reader)
    os.set_blocking(writer.fileno(), False)
    worker = object.__new__(WorkerProc)
    worker.rank = 0
    worker._external_load_completion_epoch = 7
    worker._external_load_completion_writer = writer
    worker._external_load_completion_max_payload_bytes = 4092
    thread_count = 8
    notices_per_thread = 32
    barrier = threading.Barrier(thread_count)
    emitted: list[bool] = []
    emitted_lock = threading.Lock()

    def emit(thread_id: int) -> None:
        barrier.wait()
        local_results = [
            worker._emit_external_load_completion_notice(
                thread_id * notices_per_thread + index,
                f"request-{thread_id}-{index}",
            )
            for index in range(notices_per_thread)
        ]
        with emitted_lock:
            emitted.extend(local_results)

    threads = [
        threading.Thread(target=emit, args=(thread_id,))
        for thread_id in range(thread_count)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()

        assert len(emitted) == thread_count * notices_per_thread
        assert all(emitted)
        received = []
        while reader.poll():
            received.append(pickle.loads(reader.recv_bytes(4092)))
        assert len(received) == len(emitted)
        assert len({notice.job_id for notice in received}) == len(emitted)
    finally:
        writer.close()
        reader.close()


def test_multiproc_executor_rejects_stale_completion_epoch():
    executor = object.__new__(MultiprocExecutor)
    executor._external_load_completion_epoch = 8
    executor._external_load_completion_buffer = deque()
    executor._external_load_completion_stats = defaultdict(int)

    executor._accept_external_load_completion_notice(
        ExternalLoadCompletionNotice(7, 0, 3, "request-3", 1.0)
    )

    assert not executor._external_load_completion_buffer
    assert executor._external_load_completion_stats == {"stale_epoch": 1}


def test_engine_core_disabled_completion_drain_is_zero_path():
    core = object.__new__(EngineCore)
    core._external_load_completion_notifications_enabled = False
    core.model_executor = MagicMock()
    core.scheduler = MagicMock()

    core._drain_external_load_completion_notifications()

    core.model_executor.assert_not_called()
    assert not core.model_executor.method_calls
    assert not core.scheduler.method_calls


def test_engine_core_drains_completion_before_normal_model_metadata():
    core = object.__new__(EngineCore)
    events = []
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    model_output = object()
    future = MagicMock()
    future.result.side_effect = lambda: events.append("future") or model_output
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.schedule.side_effect = (
        lambda: events.append("schedule") or scheduler_output
    )
    core.scheduler.get_grammar_bitmask.return_value = None
    core.scheduler.update_from_output.side_effect = (
        lambda *_: events.append("ordinary_metadata") or {}
    )
    core.model_executor = MagicMock()
    core.model_executor.execute_model.return_value = future
    core._drain_external_load_completion_notifications = MagicMock(
        side_effect=lambda: events.append("drain")
    )
    core._process_aborts_queue = MagicMock(
        side_effect=lambda: events.append("aborts")
    )
    core.log_error_detail = lambda *_: nullcontext()
    core.log_iteration_details = lambda *_: nullcontext()

    core.step()

    assert events == [
        "drain",
        "schedule",
        "future",
        "aborts",
        "drain",
        "ordinary_metadata",
    ]


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
