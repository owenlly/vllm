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
from vllm.v1.kv_offload.tiering.fs.common import FusedFsLoadStoreSpec
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
        dedup_before_primary: bool = False,
        fused_secondary_load: bool = False,
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
            dedup_before_primary: Ask the secondary tiers which keys they
                already hold *before* the primary tier decides what to store,
                so blocks that are already durable are never copied from the
                GPU. Only tiers whose filter_store_keys answers from local
                metadata benefit; the default keeps the original behavior.
            fused_secondary_load: Answer a secondary-tier hit with "ready"
                instead of "retry later", and let the worker do the disk read
                and the GPU copy in one job. This removes the scheduler step
                a promotion otherwise waits for. Requires every secondary tier
                to expose fused_load_paths(); the default keeps the original
                two-step promotion.
        """
        if not isinstance(profile_block_lookup, bool):
            raise TypeError("profile_block_lookup must be a boolean")
        if not isinstance(profile_job_breakdown, bool):
            raise TypeError("profile_job_breakdown must be a boolean")
        if not isinstance(dedup_before_primary, bool):
            raise TypeError("dedup_before_primary must be a boolean")
        if not isinstance(fused_secondary_load, bool):
            raise TypeError("fused_secondary_load must be a boolean")
        self.primary_tier: CPUOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []
        self.profile_block_lookup = profile_block_lookup
        self.profile_job_breakdown = profile_job_breakdown
        self.dedup_before_primary = dedup_before_primary
        self.fused_secondary_load = fused_secondary_load
        if fused_secondary_load:
            for tier in self.secondary_tiers:
                if not hasattr(tier, "fused_load_paths"):
                    raise TypeError(
                        "fused_secondary_load requires every secondary tier to "
                        f"implement fused_load_paths(); {type(tier).__name__} "
                        "does not"
                    )

        # Blocks whose primary slot was reserved at lookup time and whose bytes
        # the worker still has to read. req_id -> {key: primary block ID}.
        self._fused_load_reservations: dict[str, dict[OffloadKey, int]] = {}
        # Reservations handed to a worker, keyed by the job's identity, so that
        # complete_load/abort_load can publish or release the primary slots.
        self._pending_fused_loads: dict[
            tuple[str, frozenset[OffloadKey]], tuple[OffloadKey, ...]
        ] = {}
        # Which tier each reserved key is read from, so prepare_load() can ask
        # the right tier for its paths.
        self._fused_load_tiers: dict[OffloadKey, SecondaryTierManager] = {}
        # Blocks already in the primary tier that lookup() called ready and
        # pinned (ref_cnt+1), so a later reservation in the same request cannot
        # evict them. req_id -> {key: primary block ID}.
        self._fused_load_resident_pins: dict[str, dict[OffloadKey, int]] = {}

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

    def _on_store_completed(self, job_metadata: JobMetadata) -> None:
        """Release the Primary source after a Primary-to-Secondary STORE."""

        self.primary_tier.complete_load(
            job_metadata.keys, job_metadata.req_context
        )

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
                    self._on_store_completed(job_metadata)
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

            if self.fused_secondary_load:
                # A slot reserved earlier in this request is still ref_cnt=-1,
                # so primary.lookup() would call it in-flight and defer the
                # request — exactly the wait fusing is meant to remove.
                reserved = self._fused_load_reservations.get(req_context.req_id)
                if reserved is not None and key in reserved:
                    self._record_tiering_lookup_result(
                        "fused_secondary_reserved_ready"
                    )
                    return True

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
                if self.fused_secondary_load:
                    # Pin it now. Answering "ready" lets the caller keep walking
                    # the prefix, and reserving a slot for a later block can
                    # evict this one before prepare_load() ever asks for it.
                    # The two-step path never hit this: a secondary hit deferred
                    # the request and the whole lookup sequence restarted.
                    self._pin_resident_for_fused_load(key, req_context)
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
                    if self.fused_secondary_load:
                        if not self._reserve_fused_load(tier, key, req_context):
                            self._record_tiering_lookup_result(
                                "secondary_hit_but_no_primary_slot"
                            )
                            return False  # primary full, block unavailable
                        # "Ready" here means the fused LOAD has a source and a
                        # reserved primary slot, so the scheduler can issue it
                        # now. The KV bytes are not in DRAM yet; the worker
                        # reads them before the GPU copy.
                        self._record_tiering_lookup_result(
                            "secondary_hit_fused_ready"
                        )
                        return True
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

    @override
    def peek(self, key: OffloadKey) -> bool | None:
        """
        Check whether a block is held by any tier, without side effects.

        No promotion is started and no lookup statistics are recorded, so
        the scheduler can probe blocks it does not intend to load (for
        example, a prefix already cached on the GPU) purely for metrics.

        Returns:
            True if the primary tier or any secondary tier holds the block,
            False if every tier answered and none holds it, or None if the
            block was not found in a tier that answered and at least one
            tier cannot answer without side effects.
        """
        if self.primary_tier.peek(key):
            return True
        unknown = False
        for tier in self.secondary_tiers:
            present = tier.peek(key)
            if present is True:
                return True
            if present is None:
                unknown = True
        return None if unknown else False

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

    def _pin_resident_for_fused_load(
        self, key: OffloadKey, req_context: ReqContext
    ) -> None:
        """Raise ref_cnt on a block lookup() just reported as ready.

        Idempotent per request: a repeated lookup of the same key must not
        stack ref_cnt, because prepare_load()/complete_load() only settle it
        once.
        """
        pins = self._fused_load_resident_pins.setdefault(req_context.req_id, {})
        if key in pins:
            return
        spec = self.primary_tier.prepare_load([key], req_context)
        if not isinstance(spec, BlockIDsLoadStoreSpec):
            raise TypeError("primary tier returned invalid LOAD metadata")
        block_ids = spec.block_ids
        if len(block_ids) != 1:
            raise RuntimeError("primary lookup-time pin returned invalid block IDs")
        pins[key] = int(block_ids[0])

    def _reserve_fused_load(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """Reserve a primary slot for a block the worker will read itself.

        Unlike _initiate_promotion this submits no secondary-tier LOAD: the
        bytes are read on the worker as part of the GPU copy. The slot is
        allocated here (ref_cnt=-1) so it cannot be evicted before the worker
        fills it, and is published in complete_load() or released in
        abort_load().

        Returns:
            True if a slot was reserved, False if the primary tier is full.
        """
        primary_write_result = self.primary_tier.prepare_store([key], req_context)
        if primary_write_result is None:
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, BlockIDsLoadStoreSpec)
        block_ids = store_spec.block_ids
        if len(block_ids) != 1:
            raise RuntimeError(
                "fused LOAD reservation returned more than one primary block"
            )
        self._record_primary_eviction(
            "fused_load", len(primary_write_result.evicted_keys)
        )
        reservations = self._fused_load_reservations.setdefault(
            req_context.req_id, {}
        )
        if key in reservations:
            raise RuntimeError("fused LOAD key was reserved twice")
        reservations[key] = int(block_ids[0])
        self._fused_load_tiers[key] = tier
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

        if not self.fused_secondary_load:
            return self.primary_tier.prepare_load(keys, req_context)

        immutable_keys = tuple(keys)
        reservations = self._fused_load_reservations.get(req_context.req_id, {})
        pins = self._fused_load_resident_pins.get(req_context.req_id, {})
        fused_keys = tuple(key for key in immutable_keys if key in reservations)
        resident_keys = tuple(
            key for key in immutable_keys if key not in reservations
        )

        # Resident blocks were pinned (ref_cnt raised) when lookup() reported
        # them ready; reserved blocks are protected at ref_cnt=-1. Neither may
        # be looked up again here: a later reservation can evict an unpinned
        # block, and primary_tier.prepare_load() asserts on a missing block.
        missing = tuple(key for key in resident_keys if key not in pins)
        if missing:
            raise RuntimeError(
                "fused LOAD has resident keys without a lookup-time pin"
            )

        block_ids = [
            reservations[key] if key in reservations else pins[key]
            for key in immutable_keys
        ]
        fused_positions = tuple(
            index
            for index, key in enumerate(immutable_keys)
            if key in reservations
        )
        fused_paths: tuple[str, ...] = ()
        if fused_keys:
            paths_by_tier: dict[SecondaryTierManager, list[OffloadKey]] = {}
            for key in fused_keys:
                paths_by_tier.setdefault(self._fused_load_tiers[key], []).append(key)
            resolved: dict[OffloadKey, str] = {}
            for tier, tier_keys in paths_by_tier.items():
                for key, path in zip(tier_keys, tier.fused_load_paths(tier_keys)):
                    resolved[key] = path
            fused_paths = tuple(resolved[key] for key in fused_keys)

        identity = (req_context.req_id, frozenset(immutable_keys))
        if identity in self._pending_fused_loads:
            raise ValueError("duplicate pending fused LOAD")
        self._pending_fused_loads[identity] = fused_keys
        for key in fused_keys:
            del reservations[key]
            self._fused_load_tiers.pop(key, None)
        if not reservations:
            self._fused_load_reservations.pop(req_context.req_id, None)
        # This LOAD now owns the resident blocks' ref_cnt; complete_load() or
        # abort_load() drops it. Anything still pinned afterwards was never
        # loaded and is released in on_request_finished().
        for key in resident_keys:
            del pins[key]
        if not pins:
            self._fused_load_resident_pins.pop(req_context.req_id, None)

        return FusedFsLoadStoreSpec(block_ids, fused_positions, fused_paths)

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
        if self.fused_secondary_load:
            self._finish_fused_load(keys, req_context, success=True)
            return
        self.primary_tier.complete_load(keys, req_context)

    @override
    def abort_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Release the primary slots a failed LOAD was holding.

        Blocks the worker was going to fill still sit at ref_cnt=-1 with no
        valid contents, so they are dropped rather than published.
        """
        if self.fused_secondary_load:
            self._finish_fused_load(keys, req_context, success=False)
            return
        self.primary_tier.complete_load(keys, req_context)

    def _finish_fused_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool,
    ) -> None:
        """Settle one fused LOAD: publish or drop what the worker was filling.

        Blocks that were already resident just drop their ref_cnt. Blocks whose
        slot this manager reserved at lookup time move from ref_cnt=-1 to
        visible on success, and are freed on failure.
        """
        immutable_keys = tuple(keys)
        identity = (req_context.req_id, frozenset(immutable_keys))
        fused_keys = self._pending_fused_loads.pop(identity, ())
        fused_set = set(fused_keys)
        resident_keys = [key for key in immutable_keys if key not in fused_set]

        if resident_keys:
            self.primary_tier.complete_load(resident_keys, req_context)
        if fused_keys:
            self.primary_tier.complete_store(fused_keys, req_context, success)
            self._record_secondary_tier_blocks(
                "load",
                "fused_published" if success else "fused_failed",
                len(fused_keys),
            )

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

        # Step 1b: Drop blocks that every durable secondary tier already holds.
        # keys_to_store drives the GPU→primary copy the caller is about to
        # issue, so filtering here (rather than when cascading in
        # complete_store) is what saves the copy itself, not just the write.
        # A key is dropped only if *all* secondary tiers report it present;
        # tiers that cannot answer keep it (filter_store_keys returns it).
        if self.dedup_before_primary and self.secondary_tiers:
            remaining: Collection[OffloadKey] = keys
            for tier in self.secondary_tiers:
                if not remaining:
                    break
                remaining = tier.filter_store_keys(remaining, req_context)
            dropped = len(keys) - len(remaining)
            if dropped:
                self._record_secondary_tier_blocks(
                    "store", "skipped_present_before_primary", dropped
                )
            if not remaining:
                # Nothing left to copy: ask the primary for an empty result
                # rather than building a store spec here.
                return self.primary_tier.prepare_store([], req_context)
            keys = remaining

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
        # A request can end between lookup() and prepare_load() (preemption, an
        # abort, or a later block missing). Slots reserved for it are still at
        # ref_cnt=-1 and would leak, so release them here.
        stranded = self._fused_load_reservations.pop(req_context.req_id, None)
        if stranded:
            self.primary_tier.complete_store(
                tuple(stranded), req_context, False
            )
            for key in stranded:
                self._fused_load_tiers.pop(key, None)
            self._record_secondary_tier_blocks(
                "load", "fused_reservation_stranded", len(stranded)
            )
        # Same for blocks pinned at lookup time that no LOAD ever claimed:
        # their ref_cnt is still raised and would pin the slot forever.
        stranded_pins = self._fused_load_resident_pins.pop(req_context.req_id, None)
        if stranded_pins:
            self.primary_tier.complete_load(tuple(stranded_pins), req_context)
            self._record_secondary_tier_blocks(
                "load", "fused_pin_stranded", len(stranded_pins)
            )

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
