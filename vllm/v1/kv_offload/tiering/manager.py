# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a block-ID-addressable primary tier and zero
or more secondary tiers (Storage, Network, etc.) to provide hierarchical KV
cache offloading. The primary can be CPU-backed or implemented by another
medium such as a DPU.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary transfers use block slots allocated
   by the primary; the primary medium is not assumed to be CPU
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_load() increments ref_cnt,
   protecting blocks from eviction until complete_load() is called
"""

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
import time

import numpy as np
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    SecondaryTierManager,
)

logger = init_logger(__name__)


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompletedPromotionTiming:
    """Timing from secondary submission through primary publication."""

    submitted_at_s: float
    primary_published_at_s: float


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    standard OffloadingManager interface. The read/write aliases retain the
    original secondary-tier-facing API for compatibility. The tiering orchestrator
    uses the standard methods so other primary managers need not define aliases.
    """

    def __init__(
        self,
        num_blocks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        enable_events: bool = False,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region
        # Compatibility aliases where read/write describes the secondary
        # tier's operation on the CPU primary buffer.
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = mmap_region.create_kv_memoryview()

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        return self._kv_memoryview

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a block-ID-addressable primary tier and
    zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
        profile_block_lookup: bool = False,
        profile_job_breakdown: bool = True,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: A block-ID-addressable primary tier manager.
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
            profile_block_lookup: Whether to record per-block lookup runtimes.
            profile_job_breakdown: Whether to correlate secondary promotions
                with the subsequent primary-to-GPU load.
        """
        if not isinstance(profile_block_lookup, bool):
            raise TypeError("profile_block_lookup must be a boolean")
        if not isinstance(profile_job_breakdown, bool):
            raise TypeError("profile_job_breakdown must be a boolean")
        self.primary_tier: CPUOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []
        self.profile_block_lookup = profile_block_lookup
        self.profile_job_breakdown = profile_job_breakdown

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in on_schedule_end().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in on_schedule_end().
        self._processed_jobs_this_step: bool = False

        # Per-request set of secondary tiers that requested REQUEST_LEVEL
        # policy. Populated in on_new_request(),
        # cleaned up in on_request_finished().
        self._request_level_tiers: defaultdict[str, set[SecondaryTierManager]] = (
            defaultdict(set)
        )

        self._secondary_tier_stats: defaultdict[str, defaultdict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self._secondary_tier_job_latency_stats: defaultdict[
            str, defaultdict[str, list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        self._secondary_tier_task_runtime_stats: defaultdict[
            str, defaultdict[str, defaultdict[str, float]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._tiering_lookup_stats: defaultdict[str, int] = defaultdict(int)
        self._tiering_lookup_runtime_stats: defaultdict[str, float] = (
            defaultdict(float)
        )
        self._primary_eviction_stats: defaultdict[str, defaultdict[str, int]] = (
            defaultdict(lambda: defaultdict(int))
        )
        # Successful secondary promotions are correlated with the subsequent
        # primary-to-GPU LOAD by request ID. If a request ever needs multiple
        # promotion jobs, its visible wait starts at the first submission and
        # ends when the last promotion becomes ready.
        self._completed_promotion_timings: dict[
            str, CompletedPromotionTiming
        ] = {}

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _record_secondary_tier_job(
        self, operation: str, status: str, block_count: int
    ) -> None:
        stats = self._secondary_tier_stats[operation]
        stats[f"{status}_jobs"] += 1
        self._record_secondary_tier_blocks(operation, status, block_count)

    def _record_secondary_tier_blocks(
        self, operation: str, status: str, block_count: int
    ) -> None:
        if block_count <= 0:
            return
        self._secondary_tier_stats[operation][f"{status}_blocks"] += block_count

    def _filter_secondary_store_keys(
        self,
        tier: SecondaryTierManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> tuple[OffloadKey, ...]:
        """Filter durable keys and account for early STORE deduplication."""
        candidate_count = len(keys)
        self._record_secondary_tier_blocks("store", "candidate", candidate_count)
        store_keys = tier.filter_store_keys(keys, req_context)
        skipped_count = candidate_count - len(store_keys)
        if skipped_count < 0:
            raise ValueError("filter_store_keys returned more keys than it received")
        self._record_secondary_tier_blocks(
            "store", "skipped_present", skipped_count
        )
        return store_keys

    def _record_secondary_tier_job_latency(
        self, operation: str, status: str, latency_s: float
    ) -> None:
        self._secondary_tier_job_latency_stats[operation][status].append(
            max(latency_s, 0.0)
        )

    def _record_secondary_tier_task_runtime(
        self,
        operation: str,
        status: str,
        *,
        task_cpu_s: float,
        task_wall_s: float,
        task_count: int,
    ) -> None:
        if task_count <= 0 and task_cpu_s <= 0.0 and task_wall_s <= 0.0:
            return
        stats = self._secondary_tier_task_runtime_stats[operation][status]
        stats["cpu_seconds"] += max(task_cpu_s, 0.0)
        stats["wall_seconds"] += max(task_wall_s, 0.0)
        stats["count"] += max(task_count, 0)

    def take_secondary_tier_stats(self) -> dict[str, dict[str, int]]:
        """Return secondary tier job counters accumulated since the last call."""
        if not self._secondary_tier_stats:
            return {}
        stats = {
            operation: dict(operation_stats)
            for operation, operation_stats in self._secondary_tier_stats.items()
        }
        self._secondary_tier_stats.clear()
        return stats

    def take_secondary_tier_job_latency_stats(
        self,
    ) -> dict[str, dict[str, list[float]]]:
        """Return secondary tier job latencies accumulated since last call."""
        if not self._secondary_tier_job_latency_stats:
            return {}
        stats = {
            operation: {
                status: list(latencies)
                for status, latencies in operation_stats.items()
            }
            for operation, operation_stats in (
                self._secondary_tier_job_latency_stats.items()
            )
        }
        self._secondary_tier_job_latency_stats.clear()
        return stats

    def take_secondary_tier_task_runtime_stats(
        self,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Return secondary tier worker task runtimes since last call."""
        if not self._secondary_tier_task_runtime_stats:
            return {}
        stats = {
            operation: {
                status: dict(runtime_stats)
                for status, runtime_stats in operation_stats.items()
            }
            for operation, operation_stats in (
                self._secondary_tier_task_runtime_stats.items()
            )
        }
        self._secondary_tier_task_runtime_stats.clear()
        return stats

    def _record_tiering_lookup_total(self) -> None:
        self._tiering_lookup_stats["total_blocks"] += 1

    def _record_tiering_lookup_result(self, result: str) -> None:
        self._tiering_lookup_stats[f"{result}_blocks"] += 1

    def _record_tiering_lookup_runtime(
        self,
        tier: str,
        *,
        wall_s: float,
        cpu_s: float,
    ) -> None:
        self._tiering_lookup_runtime_stats[f"{tier}_wall_seconds"] += max(
            wall_s, 0.0
        )
        self._tiering_lookup_runtime_stats[f"{tier}_cpu_seconds"] += max(cpu_s, 0.0)

    def take_tiering_lookup_stats(self) -> dict[str, int | float]:
        """Return tiering lookup counters accumulated since the last call."""
        if not self._tiering_lookup_stats and not self._tiering_lookup_runtime_stats:
            return {}
        stats: dict[str, int | float] = dict(self._tiering_lookup_stats)
        stats.update(self._tiering_lookup_runtime_stats)
        self._tiering_lookup_stats.clear()
        self._tiering_lookup_runtime_stats.clear()
        return stats

    def _record_primary_eviction(self, cause: str, block_count: int) -> None:
        if block_count <= 0:
            return
        stats = self._primary_eviction_stats[cause]
        stats["events"] += 1
        stats["blocks"] += block_count

    def take_primary_eviction_stats(self) -> dict[str, dict[str, int]]:
        """Return primary tier eviction counters accumulated since last call."""
        if not self._primary_eviction_stats:
            return {}
        stats = {
            cause: dict(cause_stats)
            for cause, cause_stats in self._primary_eviction_stats.items()
        }
        self._primary_eviction_stats.clear()
        return stats

    def get_usage_stats(self) -> tuple[int, int]:
        """Return primary CPU tier usage for existing CPU offload metrics."""
        return self.primary_tier.get_usage_stats()

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in on_schedule_end() at the end of each step.
        """
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _on_promotion_completed(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool,
    ) -> None:
        """Hook invoked after a secondary-to-primary promotion completes."""

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_load()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_store()
           to make blocks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                completed_at_s = time.monotonic()
                job_id = completed_job.job_id
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:
                    operation = "promotion"
                    status = "succeeded" if completed_job.success else "failed"
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.complete_store(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                    self._on_promotion_completed(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                    # Primary publication is reached only after
                    # complete_store() and the completion hook publish the
                    # promoted blocks. Keep completed_at_s above for secondary
                    # job runtime, but do not report it as the request-visible
                    # publication boundary.
                    primary_published_at_s = (
                        time.monotonic()
                        if completed_job.success
                        and self.profile_job_breakdown
                        else None
                    )
                    self._record_secondary_tier_job(
                        operation,
                        status,
                        len(job_metadata.keys),
                    )
                    if primary_published_at_s is not None:
                        req_id = job_metadata.req_context.req_id
                        previous = self._completed_promotion_timings.get(req_id)
                        submitted_at_s = job_metadata.submitted_at_s
                        if previous is not None:
                            submitted_at_s = min(
                                submitted_at_s,
                                previous.submitted_at_s,
                            )
                        self._completed_promotion_timings[req_id] = (
                            CompletedPromotionTiming(
                                submitted_at_s=submitted_at_s,
                                primary_published_at_s=(
                                    primary_published_at_s
                                ),
                            )
                        )
                    else:
                        self._completed_promotion_timings.pop(
                            job_metadata.req_context.req_id,
                            None,
                        )
                else:
                    operation = "store"
                    status = "succeeded" if completed_job.success else "failed"
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    self.primary_tier.complete_load(
                        job_metadata.keys, job_metadata.req_context
                    )
                    self._record_secondary_tier_job(
                        operation,
                        status,
                        len(job_metadata.keys),
                    )
                self._record_secondary_tier_job_latency(
                    operation,
                    status,
                    completed_at_s - job_metadata.submitted_at_s,
                )
                self._record_secondary_tier_task_runtime(
                    operation,
                    status,
                    task_cpu_s=completed_job.task_cpu_s,
                    task_wall_s=completed_job.task_wall_s,
                    task_count=completed_job.task_count,
                )

    def take_completed_promotion_timing(
        self, req_id: str
    ) -> CompletedPromotionTiming | None:
        """Consume timing for a request promoted from secondary storage."""

        return self._completed_promotion_timings.pop(req_id, None)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            True  — block is ready in the primary tier.
            None  — block found but not yet ready (primary in-flight,
                    promotion started, or a secondary tier is busy).
            False — block not found in any tier, or primary is full
                    and cannot accept a promotion.
        """
        self._record_tiering_lookup_total()
        profile_lookup = self.profile_block_lookup
        total_wall_start = time.perf_counter() if profile_lookup else 0.0
        total_cpu_start = time.thread_time() if profile_lookup else 0.0
        try:
            self._maybe_process_finished_jobs()

            primary_wall_start = time.perf_counter() if profile_lookup else 0.0
            primary_cpu_start = time.thread_time() if profile_lookup else 0.0
            try:
                primary_hit = self.primary_tier.lookup(key, req_context)
            finally:
                if profile_lookup:
                    self._record_tiering_lookup_runtime(
                        "primary",
                        wall_s=time.perf_counter() - primary_wall_start,
                        cpu_s=time.thread_time() - primary_cpu_start,
                    )

            self._record_tiering_lookup_result("cpu_lookup")
            if primary_hit is True:
                self._record_tiering_lookup_result("cpu_hit_ready")
                return True
            if primary_hit is None:
                self._record_tiering_lookup_result("cpu_hit_pending")
                return None

            any_none = False
            for tier in self.secondary_tiers:
                secondary_wall_start = (
                    time.perf_counter() if profile_lookup else 0.0
                )
                secondary_cpu_start = (
                    time.thread_time() if profile_lookup else 0.0
                )
                try:
                    result = tier.lookup(key, req_context)
                finally:
                    if profile_lookup:
                        self._record_tiering_lookup_runtime(
                            "secondary",
                            wall_s=time.perf_counter() - secondary_wall_start,
                            cpu_s=time.thread_time() - secondary_cpu_start,
                        )

                self._record_tiering_lookup_result("secondary_lookup")
                if result is True:
                    if not self._initiate_promotion(tier, key, req_context):
                        self._record_tiering_lookup_result(
                            "secondary_hit_but_no_primary_slot"
                        )
                        return False  # primary full, block unavailable
                    self._record_tiering_lookup_result("secondary_hit_ready")
                    return None  # promotion started, retry later
                if result is None:
                    self._record_tiering_lookup_result("secondary_hit_pending")
                    any_none = True

            if any_none:
                return None
            self._record_tiering_lookup_result("miss")
            return False
        finally:
            if profile_lookup:
                self._record_tiering_lookup_runtime(
                    "total",
                    wall_s=time.perf_counter() - total_wall_start,
                    cpu_s=time.thread_time() - total_cpu_start,
                )

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_store().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        primary_write_result = self.primary_tier.prepare_store([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, BlockIDsLoadStoreSpec)
        self._record_primary_eviction(
            "promotion", len(primary_write_result.evicted_keys)
        )
        # Defer submit_load to on_schedule_end(). Group by (tier, request) so
        # each request's blocks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from on_schedule_end() at the end of each scheduler step,
        flushing all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                    submitted_at_s=time.monotonic(),
                )
                self._transfer_jobs[job_id] = job_metadata
                tier.submit_load(job_metadata)
                self._record_secondary_tier_job(
                    "promotion", "submitted", len(entry.keys)
                )

        self._pending_load_submissions.clear()

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # Process completed promotions to ensure blocks are ready
        self._maybe_process_finished_jobs()

        return self.primary_tier.prepare_load(keys, req_context)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        self.primary_tier.complete_load(keys, req_context)

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None
        self._record_primary_eviction("store", len(primary_result.evicted_keys))

        # Step 3: For request-level tiers, cascade blocks already in primary
        request_level_tiers = self._request_level_tiers.get(req_context.req_id)
        if request_level_tiers is not None:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.
        """
        # Filter out keys that are not ready in primary (e.g. in-flight)
        ready_keys = tuple(
            k for k in keys if self.primary_tier.lookup(k, req_context) is True
        )
        if not ready_keys:
            return

        for tier in request_level_tiers:
            store_keys = self._filter_secondary_store_keys(
                tier, ready_keys, req_context
            )
            if not store_keys:
                continue
            primary_blocks_spec = self.primary_tier.prepare_load(
                store_keys, req_context
            )

            job_id = self._next_job_id()
            assert isinstance(primary_blocks_spec, BlockIDsLoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=store_keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
                submitted_at_s=time.monotonic(),
            )
            self._transfer_jobs[job_id] = job_metadata
            tier.submit_store(job_metadata)
            self._record_secondary_tier_job(
                "store", "submitted", len(store_keys)
            )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Filter out keys known to be present before pinning primary slots
        2. Call primary.prepare_load() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        3. Call tier.submit_store() to start async transfer: primary→secondary
        4. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_load().
        """
        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if not success:
            # If GPU→Primary transfer failed, don't cascade to secondary tiers
            return

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, filter known hits before calling
        # primary.prepare_load(). This avoids pinning primary blocks for
        # secondary data that is already durable. One prepare_load() call per
        # secondary tier that still has keys to store.
        for tier in self.secondary_tiers:
            store_keys = self._filter_secondary_store_keys(tier, keys, req_context)
            if not store_keys:
                continue
            primary_blocks_spec = self.primary_tier.prepare_load(
                store_keys, req_context
            )

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()

            # Track this store job
            assert isinstance(primary_blocks_spec, BlockIDsLoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=store_keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
                submitted_at_s=time.monotonic(),
            )
            self._transfer_jobs[job_id] = job_metadata

            tier.submit_store(job_metadata)
            self._record_secondary_tier_job(
                "store", "submitted", len(store_keys)
            )

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        for tier in self.secondary_tiers:
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                self._request_level_tiers[req_context.req_id].add(tier)

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if req_context.req_id in self._request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self.primary_tier.on_request_finished(req_context)
        for tier in self.secondary_tiers:
            tier.on_request_finished(req_context)
        self._request_level_tiers.pop(req_context.req_id, None)
        self._completed_promotion_timings.pop(req_context.req_id, None)

    @override
    def on_schedule_end(self) -> None:
        """End-of-schedule hook: process finished jobs, flush deferred
        promotions, and reset the per-step gate.

        Called once per scheduler step from
        OffloadingConnectorScheduler.build_connector_meta().
        """
        self._maybe_process_finished_jobs()
        self._processed_jobs_this_step = False
        self._flush_pending_promotions()
        for tier in self.secondary_tiers:
            tier.on_schedule_end()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Yield offloading events collected since the last call.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()

        yield from self.primary_tier.take_events()

    @override
    def reset_cache(self) -> None:
        """Safely reset primary and secondary tier cache state.

        Promotions accumulated during the current scheduler step have not been
        submitted yet, so they can be failed locally. Submitted secondary jobs
        must be quiesced and reported through ``get_finished_jobs()`` before the
        primary block IDs they reference can be reused.
        """
        for pending_by_ctx in self._pending_load_submissions.values():
            for entry in pending_by_ctx.values():
                self.primary_tier.complete_store(
                    entry.keys,
                    entry.req_context,
                    success=False,
                )
        self._pending_load_submissions.clear()

        for tier in self.secondary_tiers:
            tier.quiesce()
        self._process_finished_jobs()

        if self._transfer_jobs:
            pending_job_ids = sorted(self._transfer_jobs)
            raise RuntimeError(
                "Cannot reset tiered cache while secondary jobs remain in flight: "
                f"{pending_job_ids}"
            )

        for tier in self.secondary_tiers:
            tier.reset_cache()
        self.primary_tier.reset_cache()
        self._completed_promotion_timings.clear()
        self._processed_jobs_this_step = False

    @override
    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
