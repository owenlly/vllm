# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Abstract interfaces and data types for the secondary tiering layer.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, RequestOffloadingContext

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

# Type alias for job IDs used in async transfer tracking
JobId = int


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: Collection[OffloadKey]
    block_ids: np.ndarray
    is_promotion: bool
    req_context: ReqContext
    submitted_at_s: float = 0.0


@dataclass
class JobResult:
    """Result of an async transfer job (successful or failed)."""

    job_id: JobId
    success: bool
    task_cpu_s: float = 0.0
    task_wall_s: float = 0.0
    task_count: int = 0


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers exchange data through primary-tier block slots. The
    primary tier is the gateway to the worker-visible KV cache, but it need
    not be CPU-backed:
      - Store: worker KV cache → primary → secondary  (cascade)
      - Load:  secondary → primary → worker KV cache  (promotion)

    IMPORTANT: Normal scheduling methods run in the Scheduler process and must
    be lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished_jobs() polls for completion. quiesce() is the
    reset-only synchronization hook and may wait for submitted jobs.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview | None,
        tier_type: str,
    ) -> None:
        """
        Args:
            offloading_spec: Offloading configuration.
            primary_kv_view: Optional scheduler-local memoryview of the
                primary tier. CPU-backed secondary implementations can use it
                for direct I/O; address-only primary tiers pass None.
            tier_type: Tier type identifier, set by SecondaryTierFactory
                from the registered tier type.
        """
        self._offloading_spec = offloading_spec
        self._primary_kv_view: memoryview | None = primary_kv_view
        self.tier_type = tier_type

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is present and ready,
            False if not found,
            or None if the block is being transferred (retry later).
        """
        pass

    def peek(self, key: OffloadKey) -> bool | None:
        """
        Check whether a block exists in this tier without side effects.

        Unlike lookup(), this must not start a promotion or change any
        eviction state. It only feeds metrics attribution.

        Args:
            key: Offload key to check.

        Returns:
            True if present, False if absent, or None if this tier cannot
            answer without side effects (the default).
        """
        return None

    def filter_store_keys(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> tuple[OffloadKey, ...]:
        """Return keys that still require a secondary-tier store.

        The tiering orchestrator calls this hook before pinning primary-tier
        blocks for a STORE job. Implementations may use scheduler-local
        presence metadata to exclude keys that are definitively present and
        ready in this tier. The hook must be lightweight and non-blocking;
        unknown or pending keys must remain in the returned collection.

        The default keeps every key, preserving the behavior of secondary
        tiers that do not maintain local presence metadata. ``submit_store``
        must still defensively handle already-present keys because callers
        may bypass this optimization or state may change before submission.
        """
        return tuple(keys)

    @abstractmethod
    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from the primary tier to this
        secondary tier.

        This method must be lightweight and non-blocking: allocate metadata
        and submit the transfer, but do NOT perform the data copy on the
        calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are valid primary-tier slots, pinned
            (ref-counted) for the duration of the transfer.

        The implementation is responsible for:
          1. Defensively filtering out blocks already present in this tier
          2. Evicting blocks if capacity is needed
          3. Allocating space in this tier
          4. Submitting the async transfer (read from primary via block_ids)

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to read from.
        """
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method must be lightweight and non-blocking: mark blocks as
        in-flight and submit the transfer, but do NOT perform the data copy
        on the calling thread.

        Preconditions (guaranteed by the framework):
          - ``job_metadata.block_ids`` are allocated primary-tier slots
            ready to receive data.

        The implementation must copy data from this tier into the
        primary-tier slots identified by ``block_ids``.

        Report completion via ``get_finished_jobs()``.

        Args:
            job_metadata: Job metadata including job_id, keys, and block_ids
                          identifying the primary-tier slots to write into.
        """
        pass

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Return all jobs (loads and stores) that completed since the last call.

        The framework uses these results to release resources and finalize
        transfers.

        Returns:
            Iterable of JobResult objects for jobs finished since the
            last call.
        """
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
            req_context: Per-request context.
        """
        return

    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext expressing this tier's preference
        for how blocks should be offloaded for this request.

        Args:
            req_context: Per-request context.
        """
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """
        Called when a request has finished.

        Args:
            req_context: per-request context.
        """
        return

    def on_schedule_end(self) -> None:
        """Called once at the end of each scheduler step.

        Secondary tiers may override this for per-step cleanup or
        deferred work submission.
        """
        return

    def quiesce(self) -> None:
        """Wait until previously submitted jobs can be reported as finished.

        Tiers with background transfers should override this hook. The default
        is appropriate for synchronous tiers and preserves compatibility with
        existing implementations. After this method returns, the orchestrator
        polls ``get_finished_jobs()`` and verifies that no jobs remain before
        reusing primary-tier slots during a cache reset.
        """
        return

    def reset_cache(self) -> None:
        """Reset tier-local cache state after outstanding jobs are quiesced.

        Persistent tiers may intentionally keep their data and use the default
        no-op. Tiers that track an in-memory index or own evictable storage can
        override this hook.
        """
        return

    def shutdown(self) -> None:
        """Release resources held by this tier (threads, connections, etc.)."""
        return
