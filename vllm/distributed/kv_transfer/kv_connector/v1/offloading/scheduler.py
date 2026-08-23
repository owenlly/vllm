# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_event_trace import (
    emit_kv_event_trace,
    encode_kv_keys,
    external_request_id,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Number of workers that reported an unsuccessful completion.
    failed_count: int = 0
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None
    # Scheduler-observed LOAD timing. The start is sampled immediately after
    # the scheduler creates/enqueues the logical LOAD job; the end is sampled
    # only after every worker reports completion, so TP loads use the critical
    # path. This is not a worker/native transport submission timestamp.
    load_job_created_at_s: float | None = None
    external_load_started_at_s: float | None = None
    # Exact number of non-null Host HBM PagedKV blocks held by this LOAD.
    # Counted from logical job creation until every TP worker completes.
    load_gpu_block_count: int = 0


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool
    external_load_contiguous_allocation: bool

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        external_load_contiguous_allocation = spec.extra_config.get(
            "external_load_contiguous_allocation", False
        )
        if not isinstance(external_load_contiguous_allocation, bool):
            raise TypeError(
                "external_load_contiguous_allocation must be a boolean"
            )

        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
            external_load_contiguous_allocation=(
                external_load_contiguous_allocation
            ),
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # Job-breakdown timestamps for the scheduler revisit interval. They remain
    # unset when profiling is disabled and do not affect scheduling decisions.
    external_ready_observed_at_s: float | None = None
    gpu_allocation_first_failed_at_s: float | None = None
    gpu_allocation_completed_at_s: float | None = None
    # The worker completion may be revisited while the request is still
    # blocked; record the queue-timeline boundary only once.
    external_load_ready_recorded: bool = False
    # Prevent the allocation-wait sample from being emitted twice when the
    # scheduler-revisit breakdown is also enabled.
    gpu_allocation_wait_recorded: bool = False
    agentkv_reported_prefix_lengths: tuple[int, ...] = ()

    def trace_keys_by_group(self) -> list[list[str]]:
        return [
            encode_kv_keys(group_state.offload_keys)
            for group_state in self.group_states
        ]

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()
        self._profile_job_breakdown = bool(
            getattr(spec, "profile_job_breakdown", True)
        )
        self._profile_scheduler_revisit = bool(
            self._profile_job_breakdown
            and getattr(spec, "profile_scheduler_revisit", False)
        )

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        self._current_batch_store_source_blocks_to_flush: defaultdict[
            int, set[int]
        ] = defaultdict(set)
        self._current_batch_store_source_jobs_to_retire: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}
        # Remaining ordinary per-rank completion records expected after a
        # direct 4-rank completion has already finalized a LOAD job.
        self._direct_load_duplicate_counts: dict[int, int] = {}
        self._completion_notify_counters: defaultdict[str, int] = defaultdict(int)
        self._completion_notify_latencies: defaultdict[
            str, list[float]
        ] = defaultdict(list)

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}
        self._external_load_stage_latency_stats: defaultdict[
            str, list[float]
        ] = defaultdict(list)
        self._external_load_job_gpu_block_counts: list[int] = []
        self._external_load_inflight_gpu_blocks = 0
        self._external_load_peak_inflight_gpu_blocks = 0
        self._external_load_max_job_gpu_blocks = 0
        self._external_load_inflight_jobs = 0
        self._external_load_peak_inflight_jobs = 0
        self._scheduler_iteration_started_at_s: float | None = None
        self._scheduler_waiting_phase_started_at_s: float | None = None
        self._gpu_block_pool: BlockPool | None = None
        self._external_load_gpu_extent_stats: defaultdict[str, int] = defaultdict(int)

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        if not self.config.external_load_contiguous_allocation:
            return
        gpu_block_pool.enable_contiguous_extent_allocation()
        self._gpu_block_pool = gpu_block_pool

    def get_external_load_gpu_block_extent_size(self) -> int | None:
        if (
            not self.config.external_load_contiguous_allocation
            or self.config.block_size_factor <= 1
        ):
            return None
        return self.config.block_size_factor

    def _record_external_load_gpu_extent_stats(
        self,
        pending_block_groups: Sequence[tuple[int, Sequence[int]]],
    ) -> None:
        """Record allocator choices and resulting GPU segment topology."""
        # Keep the default-off path free of block-ID traversal.
        if not self.config.external_load_contiguous_allocation:
            return
        extent_size = self.config.block_size_factor
        if extent_size <= 1:
            return

        requested_entries = 0
        contiguous_entries = 0
        gpu_segments = 0
        for logical_start_block, block_ids in pending_block_groups:
            leading_blocks = min(
                len(block_ids), (-logical_start_block) % extent_size
            )
            cursor = leading_blocks
            while cursor + extent_size <= len(block_ids):
                entry_block_ids = block_ids[cursor : cursor + extent_size]
                segments = 1 + sum(
                    current != previous + 1
                    for previous, current in zip(
                        entry_block_ids, entry_block_ids[1:]
                    )
                )
                requested_entries += 1
                contiguous_entries += segments == 1
                gpu_segments += segments
                cursor += extent_size

        stats = self._external_load_gpu_extent_stats
        stats["requested_entries"] += requested_entries
        stats["contiguous_entries"] += contiguous_entries
        stats["noncontiguous_entries"] += (
            requested_entries - contiguous_entries
        )
        stats["gpu_segments"] += gpu_segments
        stats["gpu_segment_savings"] += (
            requested_entries * extent_size - gpu_segments
        )

        if self._gpu_block_pool is None:
            raise RuntimeError("external LOAD extent allocator has no GPU block pool")
        allocator_requested, allocator_selected = (
            self._gpu_block_pool.take_contiguous_extent_allocation_stats()
        )
        if allocator_requested != requested_entries:
            raise RuntimeError(
                "external LOAD extent accounting mismatch: "
                f"allocator={allocator_requested}, load={requested_entries}"
            )
        stats["allocator_selected_entries"] += allocator_selected
        stats["allocator_fallback_entries"] += (
            allocator_requested - allocator_selected
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _start_external_load_gpu_block_tracking(
        self, gpu_block_count: int
    ) -> None:
        if gpu_block_count < 0:
            raise ValueError("external LOAD GPU block count must be non-negative")
        self._external_load_job_gpu_block_counts.append(gpu_block_count)
        self._external_load_inflight_gpu_blocks += gpu_block_count
        self._external_load_peak_inflight_gpu_blocks = max(
            self._external_load_peak_inflight_gpu_blocks,
            self._external_load_inflight_gpu_blocks,
        )
        self._external_load_max_job_gpu_blocks = max(
            self._external_load_max_job_gpu_blocks,
            gpu_block_count,
        )
        self._external_load_inflight_jobs += 1
        self._external_load_peak_inflight_jobs = max(
            self._external_load_peak_inflight_jobs,
            self._external_load_inflight_jobs,
        )

    def _finish_external_load_gpu_block_tracking(
        self, gpu_block_count: int
    ) -> None:
        if gpu_block_count < 0:
            raise ValueError("external LOAD GPU block count must be non-negative")
        self._external_load_inflight_gpu_blocks -= gpu_block_count
        self._external_load_inflight_jobs -= 1
        assert self._external_load_inflight_gpu_blocks >= 0
        assert self._external_load_inflight_jobs >= 0

    def _start_load_job_created_to_gpu_ready_timing(
        self,
        req_id: str,
        prepare_load_started_at_s: float,
        prepare_load_completed_at_s: float,
    ) -> tuple[float, float | None]:
        assert getattr(self, "_profile_job_breakdown", True)
        load_job_created_at_s = time.monotonic()
        external_load_started_at_s: float | None = None
        take_promotion_timing = getattr(
            self.manager,
            "take_completed_promotion_timing",
            None,
        )
        if callable(take_promotion_timing):
            promotion_timing = take_promotion_timing(req_id)
            if promotion_timing is not None:
                external_load_started_at_s = promotion_timing.submitted_at_s
                self._external_load_stage_latency_stats[
                    "secondary_submit_to_primary_published"
                ].append(
                    max(
                        promotion_timing.primary_published_at_s
                        - promotion_timing.submitted_at_s,
                        0.0,
                    )
                )
                self._external_load_stage_latency_stats["scheduler_gap"].append(
                    max(
                        load_job_created_at_s
                        - promotion_timing.primary_published_at_s,
                        0.0,
                    )
                )
                revisit_wait_s = max(
                    prepare_load_started_at_s
                    - promotion_timing.primary_published_at_s,
                    0.0,
                )
                self._external_load_stage_latency_stats[
                    "scheduler_revisit_wait"
                ].append(revisit_wait_s)
                if getattr(self, "_profile_scheduler_revisit", False):
                    self._record_scheduler_revisit_breakdown(
                        req_id,
                        promotion_timing.primary_published_at_s,
                        prepare_load_started_at_s,
                        revisit_wait_s,
                    )
                self._external_load_stage_latency_stats[
                    "scheduler_prepare_load"
                ].append(
                    max(
                        prepare_load_completed_at_s - prepare_load_started_at_s,
                        0.0,
                    )
                )
                self._external_load_stage_latency_stats[
                    "scheduler_build_load_spec"
                ].append(
                    max(
                        load_job_created_at_s - prepare_load_completed_at_s,
                        0.0,
                    )
                )
        return load_job_created_at_s, external_load_started_at_s

    def _record_scheduler_revisit_breakdown(
        self,
        req_id: str,
        ready_at_s: float,
        prepare_load_started_at_s: float,
        revisit_wait_s: float,
    ) -> None:
        """Record optional fine-grained scheduler revisit latency stages."""
        req_status = self._req_status.get(req_id)
        ready_observed_at_s = (
            req_status.external_ready_observed_at_s
            if req_status is not None
            else None
        )
        allocation_failed_at_s = (
            req_status.gpu_allocation_first_failed_at_s
            if req_status is not None
            else None
        )
        allocation_completed_at_s = (
            req_status.gpu_allocation_completed_at_s
            if req_status is not None
            else None
        )
        if req_status is not None:
            self._record_gpu_block_allocation_wait(req_status)
        ready_detection_wait_s = (
            max(
                min(ready_observed_at_s, prepare_load_started_at_s) - ready_at_s,
                0.0,
            )
            if ready_observed_at_s is not None
            else 0.0
        )
        iteration_started_at_s = self._scheduler_iteration_started_at_s
        waiting_phase_started_at_s = self._scheduler_waiting_phase_started_at_s
        detection_end_at_s = (
            min(ready_observed_at_s, prepare_load_started_at_s)
            if ready_observed_at_s is not None
            else ready_at_s
        )
        detection_cursor_at_s = ready_at_s
        ready_to_iteration_start_s = 0.0
        if iteration_started_at_s is not None:
            iteration_boundary_at_s = min(
                max(iteration_started_at_s, detection_cursor_at_s),
                detection_end_at_s,
            )
            ready_to_iteration_start_s = max(
                iteration_boundary_at_s - detection_cursor_at_s, 0.0
            )
            detection_cursor_at_s = iteration_boundary_at_s

        iteration_to_waiting_phase_s = 0.0
        if waiting_phase_started_at_s is not None:
            waiting_boundary_at_s = min(
                max(waiting_phase_started_at_s, detection_cursor_at_s),
                detection_end_at_s,
            )
            iteration_to_waiting_phase_s = max(
                waiting_boundary_at_s - detection_cursor_at_s, 0.0
            )
            detection_cursor_at_s = waiting_boundary_at_s

        waiting_queue_scan_s = max(
            detection_end_at_s - detection_cursor_at_s, 0.0
        )
        gpu_block_wait_s = (
            max(
                min(allocation_completed_at_s, prepare_load_started_at_s)
                - max(
                    allocation_failed_at_s,
                    (
                        ready_observed_at_s
                        if ready_observed_at_s is not None
                        else ready_at_s
                    ),
                ),
                0.0,
            )
            if allocation_failed_at_s is not None
            and allocation_completed_at_s is not None
            else 0.0
        )
        allocation_to_prepare_wait_s = (
            max(prepare_load_started_at_s - allocation_completed_at_s, 0.0)
            if allocation_completed_at_s is not None
            else 0.0
        )
        accounted_wait_s = (
            ready_detection_wait_s
            + gpu_block_wait_s
            + allocation_to_prepare_wait_s
        )
        stats = self._external_load_stage_latency_stats
        stats["scheduler_ready_detection_wait"].append(ready_detection_wait_s)
        stats["scheduler_ready_to_iteration_start"].append(
            ready_to_iteration_start_s
        )
        stats["scheduler_iteration_to_waiting_phase"].append(
            iteration_to_waiting_phase_s
        )
        stats["scheduler_waiting_queue_scan"].append(waiting_queue_scan_s)
        stats["scheduler_alloc_to_prepare_load"].append(
            allocation_to_prepare_wait_s
        )
        stats["scheduler_other_wait"].append(
            max(revisit_wait_s - accounted_wait_s, 0.0)
        )

    def _record_load_gpu_ready(
        self, job_status: TransferJobStatus
    ) -> None:
        if not getattr(self, "_profile_job_breakdown", True):
            return
        completed_at_s = time.monotonic()
        assert job_status.load_job_created_at_s is not None
        # This scheduler-side interval starts when the logical LOAD job has
        # been created/enqueued, not when a worker/native transport submits an
        # RDMA operation. It includes dispatch/worker queueing plus the TP
        # critical-path transfer through GPU readiness.
        self._external_load_stage_latency_stats[
            "scheduler_job_created_to_gpu_ready"
        ].append(
            max(
                completed_at_s - job_status.load_job_created_at_s,
                0.0,
            )
        )
        if job_status.external_load_started_at_s is not None:
            self._external_load_stage_latency_stats[
                "secondary_submit_to_gpu_ready"
            ].append(
                max(completed_at_s - job_status.external_load_started_at_s, 0.0)
            )

    def _record_gpu_block_allocation_wait(
        self, req_status: RequestOffloadState
    ) -> None:
        """Record the request's actual GPU-block allocation wait once.

        This interval is independent of the connector's promotion timing.
        Fused DPU LOADs can complete through a DPU-specific path without
        exposing a standard promotion timestamp, but they still use the
        common scheduler allocation failure/success boundary.
        """
        if not getattr(self, "_profile_job_breakdown", True):
            return
        if getattr(req_status, "gpu_allocation_wait_recorded", False):
            return
        failed_at_s = req_status.gpu_allocation_first_failed_at_s
        completed_at_s = req_status.gpu_allocation_completed_at_s
        if failed_at_s is None or completed_at_s is None:
            return
        self._external_load_stage_latency_stats[
            "scheduler_gpu_block_wait"
        ].append(max(completed_at_s - failed_at_s, 0.0))
        req_status.gpu_allocation_wait_recorded = True

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    @staticmethod
    def _store_source_block_ids(job_status: TransferJobStatus) -> set[int]:
        return set(job_status.non_sliding_window_block_ids or ()) | set(
            job_status.sliding_window_block_ids or ()
        )

    def _release_store_source_fence(self, job_id: int, block_id: int) -> None:
        """Stop tracking one source block after its fence is scheduled.

        The worker cannot enter the model forward until the fence succeeds. If
        it fails, execution terminates rather than reusing the source block.
        """

        job_status = self._jobs[job_id]
        for block_ids in (
            job_status.non_sliding_window_block_ids,
            job_status.sliding_window_block_ids,
        ):
            if block_ids is not None:
                block_ids[:] = [bid for bid in block_ids if bid != block_id]

        pending = self._block_id_to_pending_jobs.get(block_id)
        if pending is not None:
            pending.discard(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[block_id]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                num_blocks = min(
                    cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        sliding_window_size_in_blocks,
                        req_status.req_context,
                    )
                if num_hit_blocks == 0:
                    return 0

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        offloaded_block_size * (start_block_idx + num_hit_blocks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        req_status.update_offload_keys()
        self._req_status[request.request_id] = req_status
        self.manager.on_request_prefix_open(
            req_context,
            tuple(
                tuple(group_state.offload_keys)
                for group_state in req_status.group_states
            ),
        )
        req_status.agentkv_reported_prefix_lengths = tuple(
            len(group_state.offload_keys)
            for group_state in req_status.group_states
        )
        emit_kv_event_trace(
            "request_open",
            request_id=request.request_id,
            external_request_id=external_request_id(request.request_id),
            num_prompt_tokens=request.num_prompt_tokens,
            num_tokens=request.num_tokens,
            keys_by_group=req_status.trace_keys_by_group(),
            offloaded_block_sizes=[
                group.offloaded_block_size
                for group in self.config.kv_group_configs
            ],
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        if req_status.transfer_jobs:
            # A preempted request can return to the waiting queue before its
            # earlier asynchronous STORE jobs have reached the scheduler-side
            # completion barrier.  Do not let that request allocate GPU blocks
            # and create a LOAD concurrently with those jobs.  Returning None
            # keeps it in the ordinary lookup-deferred path; the scheduler will
            # revisit it after connector completions drain transfer_jobs.
            logger.debug(
                "Delaying external lookup for request %s until pending "
                "transfer jobs %s finish",
                request.request_id,
                sorted(req_status.transfer_jobs),
            )
            return None, False
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens = self._lookup(req_status)
        if num_hit_tokens is not None:
            on_lookup_resolved = getattr(
                self.manager, "on_lookup_resolved", None
            )
            if callable(on_lookup_resolved):
                on_lookup_resolved(req_status.req_context)
        if (
            num_hit_tokens
            and getattr(self, "_profile_scheduler_revisit", False)
            and req_status.external_ready_observed_at_s is None
        ):
            req_status.external_ready_observed_at_s = time.monotonic()
        req_status.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

        if num_hit_tokens is not None:
            emit_kv_event_trace(
                "request_lookup",
                request_id=request.request_id,
                num_tokens=request.num_tokens,
                num_local_tokens=num_computed_tokens,
                num_external_tokens=num_hit_tokens,
                keys_by_group=req_status.trace_keys_by_group(),
            )

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def on_external_load_allocation_failure(self, request: Request) -> None:
        if not getattr(self, "_profile_job_breakdown", True):
            return
        req_status = self._req_status.get(request.request_id)
        if (
            req_status is not None
            and req_status.gpu_allocation_first_failed_at_s is None
        ):
            req_status.gpu_allocation_first_failed_at_s = time.monotonic()

    def on_scheduler_iteration_start(self) -> None:
        if not getattr(self, "_profile_scheduler_revisit", False):
            return
        self._scheduler_iteration_started_at_s = time.monotonic()
        self._scheduler_waiting_phase_started_at_s = None

    def on_scheduler_waiting_phase_start(self) -> None:
        if not getattr(self, "_profile_scheduler_revisit", False):
            return
        self._scheduler_waiting_phase_started_at_s = time.monotonic()

    def take_external_lookup_ready_request_ids(self) -> set[str]:
        take_ready = getattr(
            self.manager,
            "take_external_lookup_ready_request_ids",
            None,
        )
        if not callable(take_ready):
            return set()
        return set(take_ready())

    def can_skip_external_lookup(
        self, request: Request, num_computed_tokens: int
    ) -> bool:
        """Whether local HBM leaves no complete offload block to query."""

        req_status = self._req_status[request.request_id]
        if req_status.transfer_jobs:
            return False

        max_hit_size_tokens = request.num_tokens
        if self._sliding_window_groups:
            max_hit_size_tokens -= 1
        return all(
            max_hit_size_tokens - num_computed_tokens
            < self.config.kv_group_configs[group_idx].offloaded_block_size
            for group_idx in self._lookup_groups
        )

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]
        if (
            getattr(self, "_profile_job_breakdown", True)
            and req_status.gpu_allocation_completed_at_s is None
        ):
            req_status.gpu_allocation_completed_at_s = time.monotonic()
            self._record_gpu_block_allocation_wait(req_status)

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        trace_load_mapping: list[dict[str, Any]] = []
        pending_block_groups: list[tuple[int, Sequence[int]]] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * gpu_block_size
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = (
                    num_locally_computed_gpu_blocks // self.config.block_size_factor
                )
                group_keys_to_load = offload_keys[start_block_idx:num_blocks]
                keys_to_load.extend(group_keys_to_load)
                for key_idx, key in enumerate(
                    group_keys_to_load, start=start_block_idx
                ):
                    gpu_start = max(
                        num_locally_computed_gpu_blocks,
                        key_idx * self.config.block_size_factor,
                    )
                    gpu_end = min(
                        num_gpu_blocks,
                        (key_idx + 1) * self.config.block_size_factor,
                    )
                    trace_load_mapping.append(
                        {
                            "key": encode_kv_keys([key])[0],
                            "gpu_block_ids": [
                                block.block_id
                                for block in group_blocks[gpu_start:gpu_end]
                                if block.block_id != 0
                            ],
                        }
                    )

            pending_block_ids = [
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            ]
            dst_block_ids.extend(pending_block_ids)
            pending_block_groups.append(
                (num_locally_computed_gpu_blocks, pending_block_ids)
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

        self._record_external_load_gpu_extent_stats(pending_block_groups)

        prepare_load_started_at_s = (
            time.monotonic() if self._profile_job_breakdown else None
        )
        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        prepare_load_completed_at_s = (
            time.monotonic() if self._profile_job_breakdown else None
        )
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )
        load_gpu_block_count = sum(
            block_id != 0 for block_id in dst_block_ids
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            transfer_spec=(src_spec, dst_spec),
        )

        if self._profile_job_breakdown:
            assert prepare_load_started_at_s is not None
            assert prepare_load_completed_at_s is not None
            (
                load_job_created_at_s,
                external_load_started_at_s,
            ) = self._start_load_job_created_to_gpu_ready_timing(
                request.request_id,
                prepare_load_started_at_s,
                prepare_load_completed_at_s,
            )
        else:
            load_job_created_at_s = None
            external_load_started_at_s = None

        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
            load_job_created_at_s=load_job_created_at_s,
            external_load_started_at_s=external_load_started_at_s,
            load_gpu_block_count=load_gpu_block_count,
        )
        emit_kv_event_trace(
            "load_submit",
            request_id=request.request_id,
            job_id=load_job_id,
            keys=encode_kv_keys(keys_to_load),
            gpu_block_ids=dst_block_ids,
            key_gpu_block_ids=trace_load_mapping,
        )
        self._start_external_load_gpu_block_tracking(load_gpu_block_count)

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()
            prefix_lengths = tuple(
                len(group_state.offload_keys)
                for group_state in req_status.group_states
            )
            if prefix_lengths != req_status.agentkv_reported_prefix_lengths:
                self.manager.on_request_prefix_update(
                    req_status.req_context,
                    tuple(
                        tuple(group_state.offload_keys)
                        for group_state in req_status.group_states
                    ),
                )
                req_status.agentkv_reported_prefix_lengths = prefix_lengths

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_block_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            block_size_factor = self.config.block_size_factor
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_block_idx * block_size_factor
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            trace_store_mapping: list[dict[str, Any]] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx
                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    trace_gpu_block_ids: list[int] = []
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        trace_gpu_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)
                    trace_store_mapping.append(
                        {
                            "key": encode_kv_keys([offload_key])[0],
                            "gpu_block_ids": trace_gpu_block_ids,
                        }
                    )

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            emit_kv_event_trace(
                "store_submit",
                request_id=req_id,
                job_id=job_id,
                keys=encode_kv_keys(keys_to_store),
                gpu_block_ids=src_block_ids,
                key_gpu_block_ids=trace_store_mapping,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, transfer_spec=(src_spec, dst_spec)
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        self.manager.on_schedule_end()

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            for job_id in req_status.transfer_jobs:
                job_status = self._jobs[job_id]
                assert job_status.is_store
                source_block_ids = self._store_source_block_ids(job_status)
                if not source_block_ids:
                    if (
                        job_status.non_sliding_window_block_ids is None
                        and job_status.sliding_window_block_ids is None
                    ):
                        # A STORE without a source map cannot use the
                        # fine-grained fence safely. Preserve whole-job wait.
                        self._current_batch_jobs_to_flush.add(job_id)
                    # Otherwise every source block was fenced by an earlier
                    # reuse/preemption and this job has no GPU dependency left.
                    continue
                self._current_batch_store_source_blocks_to_flush[job_id].update(
                    source_block_ids
                )
                for block_id in source_block_ids:
                    self._release_store_source_fence(job_id, block_id)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            reused_sources = [
                (job_id, block_id)
                for block_id in self._current_batch_allocated_block_ids
                for job_id in tuple(
                    self._block_id_to_pending_jobs.get(block_id, ())
                )
            ]
            for job_id, block_id in reused_sources:
                self._current_batch_store_source_blocks_to_flush[job_id].add(
                    block_id
                )
                self._release_store_source_fence(job_id, block_id)

        # If all tracked requests are finished, flush all pending jobs
        # (both store and load) - there might not be a future scheduler
        # step to trigger their completion.
        if self._req_status and all(
            rs.req.is_finished() for rs in self._req_status.values()
        ):
            self._current_batch_jobs_to_flush.update(self._jobs.keys())

        # Whole-job waits subsume source-only waits for the same STORE.
        for job_id in self._current_batch_jobs_to_flush:
            self._current_batch_store_source_blocks_to_flush.pop(job_id, None)

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
            store_source_blocks_to_flush={
                job_id: tuple(sorted(block_ids))
                for job_id, block_ids in (
                    self._current_batch_store_source_blocks_to_flush.items()
                )
            },
            store_source_jobs_to_retire=(
                self._current_batch_store_source_jobs_to_retire
            ),
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_store_source_blocks_to_flush = defaultdict(set)
        self._current_batch_store_source_jobs_to_retire = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        if not hasattr(self, "_direct_load_duplicate_counts"):
            self._direct_load_duplicate_counts = {}
        if not hasattr(self, "_completion_notify_counters"):
            self._completion_notify_counters = defaultdict(int)
            self._completion_notify_latencies = defaultdict(list)
        if not hasattr(self, "_current_batch_store_source_jobs_to_retire"):
            self._current_batch_store_source_jobs_to_retire = set()
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                self._direct_load_duplicate_counts.pop(job_id, None)
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            duplicate_count = self._direct_load_duplicate_counts.get(job_id)
            if duplicate_count is not None:
                self._completion_notify_counters[
                    "normal_duplicate_rank_records"
                ] += count
                remaining = duplicate_count - count
                if remaining > 0:
                    self._direct_load_duplicate_counts[job_id] = remaining
                else:
                    self._direct_load_duplicate_counts.pop(job_id, None)
                    self._completion_notify_counters[
                        "normal_duplicate_jobs"
                    ] += 1
                continue
            job_status = self._jobs[job_id]
            job_status.failed_count += meta.failed_jobs.get(job_id, 0)
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            success = job_status.failed_count == 0
            if job_status.is_store:
                self.manager.complete_store(
                    job_status.keys,
                    req_status.req_context,
                    success=success,
                )
                emit_kv_event_trace(
                    "store_complete",
                    request_id=job_status.req_id,
                    job_id=job_id,
                    success=success,
                    keys=encode_kv_keys(job_status.keys),
                )
                self._current_batch_store_source_jobs_to_retire.add(job_id)
            else:
                if not req_status.external_load_ready_recorded:
                    req_status.req.record_event(
                        EngineCoreEventType.EXTERNAL_LOAD_READY
                    )
                    req_status.external_load_ready_recorded = True
                if success:
                    self._record_load_gpu_ready(job_status)
                    self.manager.complete_load(
                        job_status.keys, req_status.req_context
                    )
                else:
                    self.manager.abort_load(
                        job_status.keys, req_status.req_context
                    )
                emit_kv_event_trace(
                    "load_complete",
                    request_id=job_status.req_id,
                    job_id=job_id,
                    success=success,
                    keys=encode_kv_keys(job_status.keys),
                )
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
                self._finish_external_load_gpu_block_tracking(
                    job_status.load_gpu_block_count
                )
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                del self._req_status[job_status.req_id]

    def can_accept_external_load_completion(
        self, job_id: int, req_id: str, worker_count: int
    ) -> bool:
        if job_id < self._stale_job_threshold:
            return False
        job = self._jobs.get(job_id)
        return bool(
            job is not None
            and not job.is_store
            and job.req_id == req_id
            and worker_count == self.config.num_workers
            # Ordinary model-step metadata may already have accounted for a
            # subset of ranks before the last direct notice is observed.  The
            # four unique success notices still prove the whole TP LOAD
            # succeeded; only the still-pending records must be synthesized.
            and 0 < job.pending_count <= worker_count
        )

    def notify_external_load_completion(
        self, job_id: int, req_id: str, worker_count: int
    ) -> int | None:
        """Finalize a successful LOAD and return normal records still due.

        The returned count is used by both scheduler layers as a job-aware
        tombstone.  ``None`` means that ordinary metadata already won, the job
        is stale, or the notice does not describe the active LOAD.
        """
        if not self.can_accept_external_load_completion(
            job_id, req_id, worker_count
        ):
            return None
        remaining = self._jobs[job_id].pending_count
        self.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=OffloadingWorkerMetadata(
                    completed_jobs={job_id: remaining}
                )
            )
        )
        self._direct_load_duplicate_counts[job_id] = remaining
        self._completion_notify_counters["fast_completed_jobs"] += 1
        return remaining

    def get_external_load_job_req_ids(
        self, job_ids: Iterable[int]
    ) -> dict[int, str]:
        return {
            job_id: job.req_id
            for job_id in job_ids
            if (job := self._jobs.get(job_id)) is not None and not job.is_store
        }

    def record_external_load_completion_notify_stats(
        self,
        counters: dict[str, int],
        latencies: dict[str, list[float]],
    ) -> None:
        for name, value in counters.items():
            if value > 0:
                self._completion_notify_counters[name] += value
        for name, samples in latencies.items():
            self._completion_notify_latencies[name].extend(
                sample for sample in samples if sample >= 0.0
            )

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        if req_status is not None:
            req_status.update_offload_keys()
            keys_by_group = req_status.trace_keys_by_group()
        else:
            keys_by_group = []
        emit_kv_event_trace(
            "request_finish",
            request_id=request.request_id,
            num_prompt_tokens=request.num_prompt_tokens,
            num_tokens=request.num_tokens,
            keys_by_group=keys_by_group,
        )

        req_context = (
            req_status.req_context if req_status else _create_req_context(request)
        )
        if req_status is not None:
            self.manager.on_request_prefix_complete(
                req_context,
                tuple(
                    tuple(group_state.offload_keys)
                    for group_state in req_status.group_states
                ),
            )
        self.manager.on_request_finished(req_context)

        if req_status is None:
            return False, None
        if not req_status.transfer_jobs:
            del self._req_status[request.request_id]
            return False, None
        # Pending stores will outlive the request's block ownership.
        # Register them so future block reuse triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=0,
                    medium=event.medium,
                    lora_name=None,
                )

    def get_kv_connector_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        if hasattr(self.manager, "get_usage_stats"):
            used_blocks, total_blocks = self.manager.get_usage_stats()
            stats.record_primary_cache_usage(
                used_blocks=used_blocks,
                total_blocks=total_blocks,
            )
        if hasattr(self.manager, "take_secondary_tier_stats"):
            secondary_tier_stats = self.manager.take_secondary_tier_stats()
            stats.record_secondary_tier_stats(secondary_tier_stats)
        if hasattr(self.manager, "take_secondary_tier_job_latency_stats"):
            latency_stats = self.manager.take_secondary_tier_job_latency_stats()
            stats.record_secondary_tier_job_latency_stats(latency_stats)
        if hasattr(self.manager, "take_secondary_tier_task_runtime_stats"):
            runtime_stats = self.manager.take_secondary_tier_task_runtime_stats()
            stats.record_secondary_tier_task_runtime_stats(runtime_stats)
        if hasattr(self.manager, "take_tiering_lookup_stats"):
            tiering_lookup_stats = self.manager.take_tiering_lookup_stats()
            stats.record_tiering_lookup_stats(tiering_lookup_stats)
        if hasattr(self.manager, "take_dpu_lookup_stage_latency_stats"):
            dpu_lookup_stage_latency_stats = (
                self.manager.take_dpu_lookup_stage_latency_stats()
            )
            stats.record_dpu_lookup_stage_latency_stats(
                dpu_lookup_stage_latency_stats
            )
        if hasattr(self.manager, "take_primary_eviction_stats"):
            primary_eviction_stats = self.manager.take_primary_eviction_stats()
            stats.record_primary_eviction_stats(primary_eviction_stats)
        if hasattr(self.manager, "take_external_load_stage_latency_stats"):
            host_latency_stats = (
                self.manager.take_external_load_stage_latency_stats()
            )
            stats.record_external_load_stage_latency_stats(host_latency_stats)
        if hasattr(self.manager, "take_fused_store_stage_latency_stats"):
            fused_store_latency_stats = (
                self.manager.take_fused_store_stage_latency_stats()
            )
            stats.record_fused_store_stage_latency_stats(
                fused_store_latency_stats
            )
        if hasattr(self.manager, "take_post_store_primary_ready_reuse_stats"):
            post_store_primary_ready_reuse_stats = (
                self.manager.take_post_store_primary_ready_reuse_stats()
            )
            stats.record_post_store_primary_ready_reuse_stats(
                post_store_primary_ready_reuse_stats
            )
        if hasattr(
            self.manager,
            "take_post_load_primary_published_cross_request_reuse_stats",
        ):
            post_load_primary_published_cross_request_reuse_stats = (
                self.manager
                .take_post_load_primary_published_cross_request_reuse_stats()
            )
            stats.record_post_load_primary_published_cross_request_reuse_stats(
                post_load_primary_published_cross_request_reuse_stats
            )
        if self._external_load_stage_latency_stats:
            stats.record_external_load_stage_latency_stats(
                {
                    stage: list(latencies)
                    for stage, latencies in (
                        self._external_load_stage_latency_stats.items()
                    )
                }
            )
            self._external_load_stage_latency_stats.clear()
        completion_counters = getattr(
            self, "_completion_notify_counters", {}
        )
        completion_latencies = getattr(
            self, "_completion_notify_latencies", {}
        )
        if completion_counters or completion_latencies:
            stats.record_external_load_completion_notify_stats(
                dict(completion_counters),
                {
                    name: list(samples)
                    for name, samples in completion_latencies.items()
                },
            )
            completion_counters.clear()
            completion_latencies.clear()
        job_gpu_blocks = getattr(
            self, "_external_load_job_gpu_block_counts", []
        )
        gpu_block_stats = {
            "job_gpu_blocks": list(job_gpu_blocks),
            "current_inflight_gpu_blocks": getattr(
                self, "_external_load_inflight_gpu_blocks", 0
            ),
            "peak_inflight_gpu_blocks": getattr(
                self, "_external_load_peak_inflight_gpu_blocks", 0
            ),
            "max_job_gpu_blocks": getattr(
                self, "_external_load_max_job_gpu_blocks", 0
            ),
            "current_inflight_jobs": getattr(
                self, "_external_load_inflight_jobs", 0
            ),
            "peak_inflight_jobs": getattr(
                self, "_external_load_peak_inflight_jobs", 0
            ),
        }
        if job_gpu_blocks or any(
            value for name, value in gpu_block_stats.items()
            if name != "job_gpu_blocks"
        ):
            stats.record_external_load_gpu_block_stats(gpu_block_stats)
            job_gpu_blocks.clear()
        external_load_gpu_extent_stats = getattr(
            self, "_external_load_gpu_extent_stats", None
        )
        if external_load_gpu_extent_stats:
            stats.record_external_load_gpu_extent_stats(
                dict(external_load_gpu_extent_stats)
            )
            external_load_gpu_extent_stats.clear()
        return None if stats.is_empty() else stats

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_store_source_blocks_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())
        self._current_batch_store_source_jobs_to_retire.update(
            job_id for job_id, job in self._jobs.items() if job.is_store
        )

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        getattr(self, "_direct_load_duplicate_counts", {}).clear()
        self._block_id_to_pending_jobs.clear()
        self._external_load_inflight_gpu_blocks = 0
        self._external_load_inflight_jobs = 0

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
