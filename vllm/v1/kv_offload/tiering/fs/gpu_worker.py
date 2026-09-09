# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side handler that fuses a filesystem read with the GPU copy.

Without this handler a block promoted from the filesystem tier crosses two
scheduler steps: ``lookup()`` starts the disk read and answers "retry later",
and only once the bytes have landed in the primary tier does a later step issue
the primary→GPU copy. The wait between those steps is pure scheduling latency —
the data is already in DRAM.

This handler collapses both stages into one worker job. The scheduler reserves
the primary slots at lookup time, answers "ready", and hands the worker a
:class:`FusedFsLoadStoreSpec` naming the files that still have to be read. The
worker reads them into its own column of the shared region and then runs the
usual primary→GPU copy, reporting one completion.
"""

import os
import time
from collections import deque

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.fs.common import FusedFsLoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)

# O_DIRECT needs the file offset, the buffer address and the length to be
# aligned. A worker's column starts at rank * cpu_page_size inside each row,
# which is not guaranteed to be aligned, so the handler probes and falls back.
_O_DIRECT_ALIGNMENT = 4096
O_DIRECT = getattr(os, "O_DIRECT", 0)


def _read_worker_column(
    source_path: str,
    view: memoryview,
    file_offset: int,
    length: int,
    use_o_direct: bool,
) -> None:
    """Read one worker's slice of a block file into its shared-region column.

    The file holds a whole block row (every worker's bytes concatenated), so
    each worker reads only ``[file_offset, file_offset + length)``. Unlike the
    scheduler-side loader this does not delete the file on failure: sibling
    ranks are reading the same file concurrently and the scheduler owns its
    lifecycle.
    """
    fd: int | None = None
    flags = os.O_RDONLY | (O_DIRECT if use_o_direct else 0)
    try:
        fd = os.open(source_path, flags)
        bytes_read = os.preadv(fd, [view], file_offset)
        if bytes_read < length:
            raise OSError(
                f"Short read from {source_path}: expected {length} bytes, "
                f"read {bytes_read}"
            )
    finally:
        if fd is not None:
            os.close(fd)


class FsGpuOffloadingHandler(OffloadingHandler):
    """Read pending blocks off the filesystem tier, then copy them to GPU.

    Wraps the existing primary→GPU handler rather than reimplementing the
    copy: once the reads land, the spec addresses the primary tier exactly
    like a ``CPULoadStoreSpec`` and is forwarded unchanged.
    """

    def __init__(
        self,
        cpu_to_gpu_handler: OffloadingHandler,
        base_view: memoryview,
        row_stride: int,
        worker_offset: int,
        worker_length: int,
        num_threads: int = 8,
    ):
        """
        Args:
            cpu_to_gpu_handler: the primary→GPU handler to forward to.
            base_view: flat byte view over the whole shared region.
            row_stride: bytes per block row (all workers concatenated).
            worker_offset: this worker's byte offset inside a row.
            worker_length: bytes this worker owns inside a row.
        """
        self._inner = cpu_to_gpu_handler
        self._base_view = base_view.cast("B")
        self._row_stride = row_stride
        self._worker_offset = worker_offset
        self._worker_length = worker_length

        self._use_o_direct = bool(O_DIRECT) and (
            worker_offset % _O_DIRECT_ALIGNMENT == 0
            and worker_length % _O_DIRECT_ALIGNMENT == 0
            and row_stride % _O_DIRECT_ALIGNMENT == 0
        )
        if not self._use_o_direct:
            logger.info(
                "Fused filesystem LOAD falls back to buffered reads: worker "
                "column (offset=%d, length=%d, row stride=%d) is not %d-byte "
                "aligned",
                worker_offset,
                worker_length,
                row_stride,
                _O_DIRECT_ALIGNMENT,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads=num_threads,
            n_write_threads=0,
            thread_name_prefix="fused_fs_load",
        )
        # job_id -> (spec forwarded once the reads land, read start time)
        self._pending_reads: dict[int, tuple[TransferSpec, float]] = {}
        # completions produced by failed reads, surfaced on the next poll
        self._failed: deque[TransferResult] = deque()
        # job_id -> seconds spent reading, merged into the inner result
        self._read_seconds: dict[int, float] = {}

    @override
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, _ = spec
        if (
            not isinstance(src_spec, FusedFsLoadStoreSpec)
            or not src_spec.fused_indices
        ):
            # Every block is already resident; this is a plain primary→GPU copy.
            return self._inner.transfer_async(job_id, spec)

        tasks = []
        for index, path in zip(src_spec.fused_indices, src_spec.fused_paths):
            block_id = int(src_spec.block_ids[index])
            start = block_id * self._row_stride + self._worker_offset
            view = self._base_view[start : start + self._worker_length]
            tasks.append(
                lambda path=path, view=view: _read_worker_column(
                    path,
                    view,
                    self._worker_offset,
                    self._worker_length,
                    self._use_o_direct,
                )
            )

        self._pending_reads[job_id] = (spec, time.perf_counter())
        self._pool.enqueue_load(job_id, len(tasks), iter(tasks))
        return True

    @override
    def get_finished(self) -> list[TransferResult]:
        # Stage 1: reads that landed become GPU copies; failures are reported.
        for result in self._pool.get_finished():
            pending = self._pending_reads.pop(result.job_id, None)
            if pending is None:
                logger.warning(
                    "Fused filesystem LOAD saw an unknown read job %d",
                    result.job_id,
                )
                continue
            spec, started_at = pending
            read_seconds = time.perf_counter() - started_at
            if not result.success:
                self._failed.append(
                    TransferResult(job_id=result.job_id, success=False)
                )
                continue
            self._read_seconds[result.job_id] = read_seconds
            if not self._inner.transfer_async(result.job_id, spec):
                self._read_seconds.pop(result.job_id, None)
                self._failed.append(
                    TransferResult(job_id=result.job_id, success=False)
                )

        # Stage 2: GPU copies that landed, annotated with their read time.
        results = list(self._inner.get_finished())
        for transfer in results:
            read_seconds = self._read_seconds.pop(transfer.job_id, None)
            if read_seconds is None:
                continue
            stages = dict(transfer.fused_load_stage_latencies or {})
            stages["secondary_to_primary"] = read_seconds
            transfer.fused_load_stage_latencies = stages

        while self._failed:
            results.append(self._failed.popleft())
        return results

    @override
    def wait(self, job_ids: set[int]) -> None:
        # Reads must land (and be forwarded) before the GPU copies can finish.
        while job_ids & self._pending_reads.keys():
            self.get_finished()
            if job_ids & self._pending_reads.keys():
                self._pool.wait_idle()
                self.get_finished()
        self._inner.wait(job_ids - {r.job_id for r in self._failed})

    @override
    def shutdown(self) -> None:
        self._pool.shutdown()
        self._inner.shutdown()
