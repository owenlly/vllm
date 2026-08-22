# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LoadStoreSpec

# a single transfer spec (src_blocks_spec, dst_blocks_spec)
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
# transfers are forwarded to workers by (src_medium, dst_medium)
TransferType = tuple[str, str]

logger = init_logger(__name__)


@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = None  # Size in bytes
    transfer_time: float | None = None
    transfer_type: TransferType | None = None
    rdma_stats: dict[str, int] | None = None
    # Optional rank-local fused LOAD stage durations, in seconds. Connector-
    # specific handlers may populate this without changing the generic
    # transfer interface for other backends.
    fused_load_stage_latencies: dict[str, float] | None = None
    # Optional connector-specific fused STORE stages, parallel to fused LOAD.
    fused_store_stage_latencies: dict[str, float] | None = None
    # Per-entry time from immutable KV availability until the rank-local GPU
    # source is no longer read. Keeping the samples avoids a job-size-biased
    # mean in connector metrics.
    fused_store_source_safe_latencies: list[float] | None = None
    fused_store_dedup_stats: dict[str, int] | None = None
    # Optional bounded resident/admission counters and byte accounting.
    fused_resident_stats: dict[str, int] | None = None


class OffloadingHandler(ABC):
    """
    OffloadingHandler class for managing asynchronous KV data transfers

    This class runs in the worker.
    It kicks off async KV data transfer requests, and allows
    collecting back completion statuses.

    The class provides the following primitives:
        transfer_async() - kicks off a new transfer job
        get_finished() - returns a list of newly finished job IDs.
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of (job_id, success) of transfers.
        """
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """

    def wait_store_source_safe(
        self, job_source_block_ids: dict[int, tuple[int, ...]]
    ) -> None:
        """Wait until STORE jobs no longer read selected GPU source blocks.

        Handlers without finer-grained source completion preserve the existing
        behavior by waiting for each complete job.
        """

        self.wait(set(job_source_block_ids))

    def retire_store_source_jobs(self, job_ids: set[int]) -> None:
        """Discard source-safety state for scheduler-retired STORE jobs."""

        return

    def shutdown(self) -> None:
        """Shutdown the handler and release any resources."""
        return


class OffloadingWorker:
    """
    OffloadingWorker class for managing asynchronous KV data transfers
    using multiple OffloadingHandlers

    This class runs in the worker.
    It kicks off async KV data transfer requests, by delegating
    to one of its registered OffloadingHandlers, based on the transfer type.

    The class provides the following primitives:
        register_handler() - registers a new handler to handle
            a specific transfer type
        transfer_async() - kicks off a new transfer job
            using one of the registered handlers.
        get_finished() - returns a list of newly finished job IDs
            from all handlers.
    """

    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}
        # STORE handlers are retained past rank-local completion so a later
        # block-reuse fence can still consult connector-specific tombstones.
        self._store_job_to_handler: dict[int, OffloadingHandler] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        """
        Registers a new handler.

        Args:
            src_cls: the source type of transfers handled by this handler.
            dst_cls: the destination type of transfers handled by this handler.
            handler: the handler that will handle transfers.
        """
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiates an asynchronous transfer of KV data.

        Args:
            job_id: a unique ID that will be used when notifying back on
                transfer completion.
            spec: the (src, dst) spec of the KV data transfer.

        Returns:
            True if transfer was submitted successfully.
        """
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None
        if src.medium() == "GPU" and job_id in self._store_job_to_handler:
            raise ValueError(f"duplicate STORE transfer job ID {job_id}")
        try:
            success = handler.transfer_async(job_id, spec)
        except Exception as e:
            logger.warning(
                "Exception in %r transfer %d: %r",
                transfer_type,
                job_id,
                e,
                exc_info=True,
            )
            return False

        if not success:
            logger.warning("Failed to submit %r transfer %d", transfer_type, job_id)
        else:
            if src.medium() == "GPU":
                self._store_job_to_handler[job_id] = handler
            logger.debug("Submitted %r transfer %d: %r", transfer_type, job_id, spec)
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Get transfers finished since last call.

        Returns:
            A list of TransferResults
        """
        finished = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        """
        Wait for jobs to finish (blocking).

        Args:
            job_ids: The set of job IDs to wait for.
        """
        for handler in self.handlers:
            handler.wait(job_ids)

    def wait_store_source_safe(
        self, job_source_block_ids: dict[int, tuple[int, ...]]
    ) -> None:
        """Route rank-local STORE source fences to the owning handlers."""

        by_handler: dict[OffloadingHandler, dict[int, tuple[int, ...]]] = {}
        for job_id, block_ids in job_source_block_ids.items():
            handler = self._store_job_to_handler.get(job_id)
            if handler is None:
                raise RuntimeError(
                    f"STORE source fence references unknown job {job_id}"
                )
            by_handler.setdefault(handler, {})[job_id] = block_ids
        for handler, requirements in by_handler.items():
            handler.wait_store_source_safe(requirements)

    def retire_store_source_jobs(self, job_ids: set[int]) -> None:
        """Retire STORE source-safety state after whole TP completion."""

        by_handler: dict[OffloadingHandler, set[int]] = {}
        for job_id in job_ids:
            handler = self._store_job_to_handler.pop(job_id, None)
            if handler is not None:
                by_handler.setdefault(handler, set()).add(job_id)
        for handler, retired_job_ids in by_handler.items():
            handler.retire_store_source_jobs(retired_job_ids)

    def shutdown(self) -> None:
        for handler in self.handlers:
            handler.shutdown()
        self._store_job_to_handler.clear()
