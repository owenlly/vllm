# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
    to_keys,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.executor.external_load_completion import (
    ExternalLoadCompletionNotice,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    OffloadingEvent,
    OffloadingManager,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offloading_connector(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # 3 blocks, store just the middle block (skip first and last)
    # blocks = [0, 1, 2], [3, 4, 5], [6, 7, 8]
    runner.new_request(token_ids=[0] * offloaded_block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys)[1:2])
    )
    runner.run(decoded_tokens=[0])

    # add block missing 1 token -> no offload
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size - 1),
        expected_stored=(3, 4, 5),
    )
    runner.manager.touch.assert_not_called()

    # +1 token -> single block, fail prepare_store
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # 1 more block (+ token for async scheduling)
    # now set block_hashes_to_store = []
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[0] * (offloaded_block_size + 1))

    # 1 more block (+ token for kicking off offloading)
    # now check touch was called with all 6 blocks
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored=(15, 16, 17),
    )
    runner.manager.touch.assert_called()
    block_hashes1 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes1) == 6

    # terminate request
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # create a new request differing only on the last token
    runner.new_request(token_ids=[0] * (offloaded_block_size * 6 - 1) + [1])
    runner.run(decoded_tokens=[0])
    runner.manager.touch.assert_called()
    block_hashes2 = list(runner.manager.touch.call_args.args[0])
    assert len(block_hashes2) == 6

    # verify hashes are the same, except for the last block
    assert block_hashes1[:5] == block_hashes2[:5]
    assert block_hashes1[5] != block_hashes2[5]

    # terminate request
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=tuple(range(6 * block_size_factor)),
    )

    # full_block_tokens - num_computed_tokens < offloaded_block_size
    runner.new_request(
        token_ids=[0] * block_size + [1] * (offloaded_block_size - block_size)
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_not_called()

    # single block lookup with no hits
    runner.new_request(token_ids=[1] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    runner.manager.lookup.assert_called_once()

    # single block lookup with a hit
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(0, 1, 2))

    # single block lookup with a hit in a middle block
    runner.new_request(
        token_ids=[0] * offloaded_block_size * 2 + [1] * offloaded_block_size
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_loaded=(3, 4, 5))

    # test take_events
    def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
        return [BlockHash(str(i).encode()) for i in int_hashes]

    def take_events() -> Iterable[OffloadingEvent]:
        yield OffloadingEvent(keys=to_keys([1, 2, 3]), medium="A", removed=False)
        yield OffloadingEvent(keys=to_keys([4, 5, 6]), medium="B", removed=True)

    runner.manager.take_events.side_effect = take_events
    events = list(runner.scheduler_connector.take_events())
    assert len(events) == 2
    event = events[0]
    assert isinstance(event, BlockStored)
    assert event.block_hashes == to_hashes([1, 2, 3])
    assert event.block_size == 0
    assert event.medium == "A"
    assert event.token_ids == []
    assert event.parent_block_hash is None
    assert event.lora_id is None
    assert event.lora_name is None
    event = events[1]
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == to_hashes([4, 5, 6])
    assert event.medium == "B"


def _fragment_oldest_gpu_blocks(runner):
    block_pool = runner.scheduler.kv_cache_manager.block_pool
    held_blocks = block_pool.get_new_blocks(16)
    block_pool.free_blocks(held_blocks[::2], prepend=True)
    return block_pool


def test_external_load_contiguous_allocation_default_off_has_no_index(
    request_runner,
):
    runner = request_runner(
        block_size=16,
        num_gpu_blocks=80,
        async_scheduling=True,
        block_size_factor=8,
    )
    block_pool = runner.scheduler.kv_cache_manager.block_pool

    assert block_pool.free_block_queue._free_block_ids is None
    assert runner.connector_scheduler.get_external_load_gpu_block_extent_size() is None


def test_confirmed_async_external_load_uses_contiguous_gpu_extent(
    request_runner,
):
    runner = request_runner(
        block_size=16,
        num_gpu_blocks=80,
        async_scheduling=True,
        block_size_factor=8,
        extra_config_overrides={
            "external_load_contiguous_allocation": True
        },
    )
    block_pool = _fragment_oldest_gpu_blocks(runner)
    runner.new_request(token_ids=[0] * 128)
    runner.connector_scheduler._maximal_prefix_lookup = (
        lambda keys, req_context: 1
    )

    runner.scheduler.schedule()

    manager = (
        runner.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    )
    block_ids = [
        block.block_id for block in manager.req_to_blocks["0"]
    ]
    assert block_ids == list(range(17, 25))
    assert block_pool.take_contiguous_extent_allocation_stats() == (0, 0)
    assert dict(runner.connector_scheduler._external_load_gpu_extent_stats) == {
        "requested_entries": 1,
        "allocator_selected_entries": 1,
        "allocator_fallback_entries": 0,
        "contiguous_entries": 1,
        "noncontiguous_entries": 0,
        "gpu_segments": 1,
        "gpu_segment_savings": 7,
    }


def test_ordinary_request_keeps_normal_gpu_allocator(request_runner):
    runner = request_runner(
        block_size=16,
        num_gpu_blocks=80,
        async_scheduling=True,
        block_size_factor=8,
        extra_config_overrides={
            "external_load_contiguous_allocation": True
        },
    )
    block_pool = _fragment_oldest_gpu_blocks(runner)
    runner.new_request(token_ids=[0] * 128)
    runner.connector_scheduler._maximal_prefix_lookup = (
        lambda keys, req_context: 0
    )

    runner.scheduler.schedule()

    manager = (
        runner.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    )
    assert [
        block.block_id for block in manager.req_to_blocks["0"]
    ] == list(range(1, 16, 2))
    assert block_pool.take_contiguous_extent_allocation_stats() == (0, 0)
    assert not runner.connector_scheduler._external_load_gpu_extent_stats


def test_pending_external_lookup_does_not_preallocate_gpu_extents(
    request_runner,
):
    runner = request_runner(
        block_size=16,
        num_gpu_blocks=80,
        async_scheduling=True,
        block_size_factor=8,
        extra_config_overrides={
            "external_load_contiguous_allocation": True
        },
    )
    block_pool = runner.scheduler.kv_cache_manager.block_pool
    free_before = block_pool.get_num_free_blocks()
    runner.new_request(token_ids=[0] * 128)
    runner.connector_scheduler._maximal_prefix_lookup = (
        lambda keys, req_context: None
    )

    runner.scheduler.schedule()

    manager = (
        runner.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    )
    assert "0" not in manager.req_to_blocks
    assert block_pool.get_num_free_blocks() == free_before
    assert block_pool.take_contiguous_extent_allocation_stats() == (0, 0)
    assert not runner.connector_scheduler._external_load_gpu_extent_stats


def test_external_load_stage_latency_tracks_request_critical_path():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = True
    scheduler.manager = MagicMock()
    scheduler.manager.take_completed_promotion_timing = MagicMock(
        return_value=SimpleNamespace(
            submitted_at_s=1.0,
            primary_published_at_s=2.0,
        )
    )
    scheduler._req_status = {
        "request": SimpleNamespace(
            external_ready_observed_at_s=2.015625,
            gpu_allocation_first_failed_at_s=2.03125,
            gpu_allocation_completed_at_s=2.046875,
        )
    }
    scheduler._scheduler_iteration_started_at_s = 2.00390625
    scheduler._scheduler_waiting_phase_started_at_s = 2.0078125
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic",
        side_effect=[2.25, 2.75],
    ):
        job_created_at_s, external_started_at_s = (
            scheduler._start_load_job_created_to_gpu_ready_timing(
                "request",
                prepare_load_started_at_s=2.0625,
                prepare_load_completed_at_s=2.125,
            )
        )
        scheduler._record_load_gpu_ready(
            TransferJobStatus(
                req_id="request",
                pending_count=0,
                keys=set(),
                is_store=False,
                load_job_created_at_s=job_created_at_s,
                external_load_started_at_s=external_started_at_s,
            )
        )
    assert scheduler._external_load_stage_latency_stats == {
        "secondary_submit_to_primary_published": [1.0],
        "scheduler_gap": [0.25],
        "scheduler_revisit_wait": [0.0625],
        "scheduler_ready_detection_wait": [0.015625],
        "scheduler_ready_to_iteration_start": [0.00390625],
        "scheduler_iteration_to_waiting_phase": [0.00390625],
        "scheduler_waiting_queue_scan": [0.0078125],
        "scheduler_gpu_block_wait": [0.015625],
        "scheduler_alloc_to_prepare_load": [0.015625],
        "scheduler_other_wait": [0.015625],
        "scheduler_prepare_load": [0.0625],
        "scheduler_build_load_spec": [0.125],
        "scheduler_job_created_to_gpu_ready": [0.5],
        "secondary_submit_to_gpu_ready": [1.75],
    }


def test_can_skip_external_lookup_requires_no_complete_offload_block():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        kv_group_configs=[SimpleNamespace(offloaded_block_size=128)]
    )
    scheduler._lookup_groups = (0,)
    scheduler._sliding_window_groups = ()
    scheduler._req_status = {"request": SimpleNamespace(transfer_jobs=set())}
    request = SimpleNamespace(request_id="request", num_tokens=256)

    assert not scheduler.can_skip_external_lookup(request, 128)
    assert scheduler.can_skip_external_lookup(request, 144)

    scheduler._req_status["request"].transfer_jobs.add(7)
    assert not scheduler.can_skip_external_lookup(request, 144)


def test_take_external_lookup_ready_request_ids_forwards_manager_hints():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = MagicMock()
    scheduler.manager.take_external_lookup_ready_request_ids.return_value = {
        "ready-b",
        "ready-a",
    }

    assert scheduler.take_external_lookup_ready_request_ids() == {
        "ready-a",
        "ready-b",
    }


def test_external_load_allocation_failure_records_first_failure_only():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = True
    req_status = SimpleNamespace(gpu_allocation_first_failed_at_s=None)
    scheduler._req_status = {"request": req_status}
    request = SimpleNamespace(request_id="request")

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic",
        side_effect=[3.0, 4.0],
    ):
        scheduler.on_external_load_allocation_failure(request)
        scheduler.on_external_load_allocation_failure(request)

    assert req_status.gpu_allocation_first_failed_at_s == 3.0


def test_external_load_allocation_failure_is_tracked_by_job_breakdown():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = False
    req_status = SimpleNamespace(gpu_allocation_first_failed_at_s=None)
    scheduler._req_status = {"request": req_status}
    request = SimpleNamespace(request_id="request")

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic",
        return_value=3.0,
    ):
        scheduler.on_external_load_allocation_failure(request)

    assert req_status.gpu_allocation_first_failed_at_s == 3.0


def test_gpu_block_allocation_wait_is_recorded_without_revisit_timing():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = False
    scheduler._external_load_stage_latency_stats = defaultdict(list)
    req_status = SimpleNamespace(
        gpu_allocation_first_failed_at_s=3.0,
        gpu_allocation_completed_at_s=4.25,
        gpu_allocation_wait_recorded=False,
    )

    scheduler._record_gpu_block_allocation_wait(req_status)
    scheduler._record_gpu_block_allocation_wait(req_status)

    assert scheduler._external_load_stage_latency_stats == {
        "scheduler_gpu_block_wait": [1.25]
    }
    assert req_status.gpu_allocation_wait_recorded is True


def test_scheduler_phase_timestamps_follow_current_iteration():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = True

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic",
        side_effect=[1.0, 1.25, 2.0],
    ):
        scheduler.on_scheduler_iteration_start()
        scheduler.on_scheduler_waiting_phase_start()
        assert scheduler._scheduler_iteration_started_at_s == 1.0
        assert scheduler._scheduler_waiting_phase_started_at_s == 1.25

        scheduler.on_scheduler_iteration_start()

    assert scheduler._scheduler_iteration_started_at_s == 2.0
    assert scheduler._scheduler_waiting_phase_started_at_s is None


def test_scheduler_revisit_breakdown_is_disabled_by_default():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = True
    scheduler._profile_scheduler_revisit = False
    scheduler.manager = MagicMock()
    scheduler.manager.take_completed_promotion_timing = MagicMock(
        return_value=SimpleNamespace(
            submitted_at_s=1.0,
            primary_published_at_s=2.0,
        )
    )
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic",
        return_value=2.25,
    ):
        scheduler._start_load_job_created_to_gpu_ready_timing(
            "request",
            prepare_load_started_at_s=2.0625,
            prepare_load_completed_at_s=2.125,
        )

    assert scheduler._external_load_stage_latency_stats == {
        "secondary_submit_to_primary_published": [1.0],
        "scheduler_gap": [0.25],
        "scheduler_revisit_wait": [0.0625],
        "scheduler_prepare_load": [0.0625],
        "scheduler_build_load_spec": [0.125],
    }


def test_external_load_stage_latency_can_be_disabled():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._profile_job_breakdown = False
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading."
        "scheduler.time.monotonic"
    ) as monotonic:
        scheduler._record_load_gpu_ready(
            TransferJobStatus(
                req_id="request",
                pending_count=0,
                keys=set(),
                is_store=False,
            )
        )

    monotonic.assert_not_called()
    assert scheduler._external_load_stage_latency_stats == {}


def test_external_load_gpu_block_tracking_records_job_and_concurrency_peaks():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._external_load_job_gpu_block_counts = []
    scheduler._external_load_inflight_gpu_blocks = 0
    scheduler._external_load_peak_inflight_gpu_blocks = 0
    scheduler._external_load_max_job_gpu_blocks = 0
    scheduler._external_load_inflight_jobs = 0
    scheduler._external_load_peak_inflight_jobs = 0

    scheduler._start_external_load_gpu_block_tracking(128)
    scheduler._start_external_load_gpu_block_tracking(512)
    scheduler._finish_external_load_gpu_block_tracking(128)

    assert scheduler._external_load_job_gpu_block_counts == [128, 512]
    assert scheduler._external_load_inflight_gpu_blocks == 512
    assert scheduler._external_load_peak_inflight_gpu_blocks == 640
    assert scheduler._external_load_max_job_gpu_blocks == 512
    assert scheduler._external_load_inflight_jobs == 1
    assert scheduler._external_load_peak_inflight_jobs == 2

    scheduler._finish_external_load_gpu_block_tracking(512)
    assert scheduler._external_load_inflight_gpu_blocks == 0
    assert scheduler._external_load_inflight_jobs == 0


def test_external_lookup_events_use_request_queue_iteration_boundaries():
    scheduler = object.__new__(Scheduler)
    scheduler.log_stats = True
    request = SimpleNamespace(
        external_lookup_started_recorded=False,
        external_lookup_finished_recorded=False,
        record_event=MagicMock(),
    )

    scheduler._record_external_lookup_started(request, 10.0)
    scheduler._record_external_lookup_started(request, 11.0)
    scheduler._record_external_lookup_finished(request, 12.0)
    scheduler._record_external_lookup_finished(request, 13.0)

    assert request.record_event.call_args_list == [
        call(EngineCoreEventType.EXTERNAL_LOOKUP_STARTED, 10.0),
        call(EngineCoreEventType.EXTERNAL_LOOKUP_FINISHED, 12.0),
    ]


def test_external_load_revisit_uses_request_queue_schedule_boundary():
    scheduler = object.__new__(Scheduler)
    scheduler.finished_recving_kv_req_ids = {"request-0"}
    scheduler._update_waiting_for_remote_kv = MagicMock()
    request = SimpleNamespace(
        request_id="request-0",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        num_preemptions=0,
        record_event=MagicMock(),
    )
    scheduled_timestamp = 123.5

    assert scheduler._try_promote_blocked_waiting_request(
        request, scheduled_timestamp
    )

    scheduler._update_waiting_for_remote_kv.assert_called_once_with(request)
    request.record_event.assert_called_once_with(
        EngineCoreEventType.EXTERNAL_LOAD_REVISITED,
        scheduled_timestamp,
    )
    assert request.status == RequestStatus.WAITING


def test_external_load_completion_records_ready_exactly_once():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.is_finished.return_value = False
    req_status = SimpleNamespace(
        req=request,
        req_context=MagicMock(),
        transfer_jobs={7},
        external_load_ready_recorded=False,
    )
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        7: TransferJobStatus(
            req_id="request-0",
            pending_count=1,
            keys=set(),
            is_store=False,
            load_gpu_block_count=32,
        )
    }
    scheduler._req_status = {"request-0": req_status}
    scheduler.manager = MagicMock()
    scheduler._record_load_gpu_ready = MagicMock()
    scheduler._blocks_being_loaded = set()
    scheduler._block_id_to_pending_jobs = {}
    scheduler._finish_external_load_gpu_block_tracking = MagicMock()

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 1}
            )
        )
    )

    request.record_event.assert_called_once_with(
        EngineCoreEventType.EXTERNAL_LOAD_READY
    )
    scheduler._finish_external_load_gpu_block_tracking.assert_called_once_with(
        32
    )


def test_offloading_direct_and_normal_completion_finalize_once():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.is_finished.return_value = False
    req_status = SimpleNamespace(
        req=request,
        req_context=MagicMock(),
        transfer_jobs={7},
        external_load_ready_recorded=False,
    )
    scheduler.config = SimpleNamespace(num_workers=1)
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        7: TransferJobStatus(
            req_id="request-0",
            pending_count=1,
            keys=set(),
            is_store=False,
            load_gpu_block_count=32,
        )
    }
    scheduler._req_status = {"request-0": req_status}
    scheduler.manager = MagicMock()
    scheduler._record_load_gpu_ready = MagicMock()
    scheduler._blocks_being_loaded = set()
    scheduler._block_id_to_pending_jobs = {}
    scheduler._finish_external_load_gpu_block_tracking = MagicMock()
    scheduler._direct_load_duplicate_counts = {}
    scheduler._completion_notify_counters = defaultdict(int)
    scheduler._completion_notify_latencies = defaultdict(list)

    assert scheduler.notify_external_load_completion(7, "request-0", 1) == 1
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 1}
            )
        )
    )

    scheduler.manager.complete_load.assert_called_once()
    request.record_event.assert_called_once_with(
        EngineCoreEventType.EXTERNAL_LOAD_READY
    )
    scheduler._finish_external_load_gpu_block_tracking.assert_called_once_with(
        32
    )
    assert scheduler._direct_load_duplicate_counts == {}


def _make_direct_completion_scheduler(
    status: RequestStatus = RequestStatus.WAITING_FOR_REMOTE_KVS,
):
    scheduler = object.__new__(Scheduler)
    connector = MagicMock()
    connector.can_accept_external_load_completion.return_value = True
    connector.notify_external_load_completion.return_value = 4
    connector.get_external_load_job_req_ids.return_value = {}
    scheduler.connector = connector
    scheduler._direct_external_load_completion_enabled = True
    scheduler.parallel_config = SimpleNamespace(world_size=4)
    request = SimpleNamespace(request_id="request-0", status=status)
    scheduler.requests = {request.request_id: request}
    scheduler.finished_recving_kv_req_ids = set()
    scheduler._external_load_completion_ranks = {}
    scheduler._direct_external_load_completed_jobs = {}
    scheduler._free_blocks = MagicMock()
    return scheduler, connector, request


def _notices(job_id: int = 7, req_id: str = "request-0"):
    return [
        ExternalLoadCompletionNotice(1, rank, job_id, req_id, 1.0)
        for rank in range(4)
    ]


def test_direct_external_load_requires_four_unique_ranks():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler.notify_external_load_completions(
        [_notices()[0], _notices()[0], *_notices()[1:]]
    )

    connector.notify_external_load_completion.assert_called_once_with(
        7, "request-0", 4
    )
    assert scheduler.finished_recving_kv_req_ids == {"request-0"}
    assert scheduler._direct_external_load_completed_jobs == {
        7: ("request-0", 4)
    }


def test_direct_completion_disabled_preserves_original_update_path():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler._direct_external_load_completion_enabled = False

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"request-0"})
    )

    assert scheduler.finished_recving_kv_req_ids == {"request-0"}
    connector.update_connector_output.assert_called_once()
    connector.get_external_load_job_req_ids.assert_not_called()


def test_direct_external_load_tracks_only_remaining_normal_records():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    connector.notify_external_load_completion.return_value = 2

    scheduler.notify_external_load_completions(_notices())

    assert scheduler._direct_external_load_completed_jobs == {
        7: ("request-0", 2)
    }


def test_partial_ordinary_metadata_keeps_direct_rank_aggregate():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    connector.get_external_load_job_req_ids.side_effect = [
        {7: "request-0"},
        {7: "request-0"},
    ]
    connector.notify_external_load_completion.return_value = 3

    scheduler.notify_external_load_completions(_notices()[:2])
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 1}
            )
        )
    )

    assert scheduler._external_load_completion_ranks[7][1] == {0, 1}

    scheduler.notify_external_load_completions(_notices()[2:])

    connector.notify_external_load_completion.assert_called_once_with(
        7, "request-0", 4
    )
    assert scheduler.finished_recving_kv_req_ids == {"request-0"}
    assert scheduler._direct_external_load_completed_jobs == {
        7: ("request-0", 3)
    }


def test_direct_completion_finishes_only_remaining_ordinary_rank_records():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.is_finished.return_value = False
    req_status = SimpleNamespace(
        req=request,
        req_context=MagicMock(),
        transfer_jobs={7},
        external_load_ready_recorded=False,
    )
    scheduler.config = SimpleNamespace(num_workers=4)
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        7: TransferJobStatus(
            req_id="request-0",
            pending_count=2,
            keys=set(),
            is_store=False,
            load_gpu_block_count=32,
        )
    }
    scheduler._req_status = {"request-0": req_status}
    scheduler.manager = MagicMock()
    scheduler._record_load_gpu_ready = MagicMock()
    scheduler._blocks_being_loaded = set()
    scheduler._block_id_to_pending_jobs = {}
    scheduler._finish_external_load_gpu_block_tracking = MagicMock()
    scheduler._direct_load_duplicate_counts = {}
    scheduler._completion_notify_counters = defaultdict(int)
    scheduler._completion_notify_latencies = defaultdict(list)

    assert scheduler.notify_external_load_completion(7, "request-0", 4) == 2
    assert scheduler._direct_load_duplicate_counts == {7: 2}
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 2}
            )
        )
    )

    scheduler.manager.complete_load.assert_called_once()
    assert scheduler._direct_load_duplicate_counts == {}


def test_direct_then_normal_metadata_does_not_repeat_lifecycle():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler.notify_external_load_completions(_notices())

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 4}
            ),
        )
    )

    scheduler._free_blocks.assert_not_called()
    assert scheduler._direct_external_load_completed_jobs == {}
    connector.update_connector_output.assert_called_once()


def test_cancelled_direct_then_normal_frees_exactly_once():
    scheduler, _, _ = _make_direct_completion_scheduler(
        RequestStatus.FINISHED_ABORTED
    )
    scheduler.notify_external_load_completions(_notices())
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 4}
            ),
        )
    )

    scheduler._free_blocks.assert_called_once()


def test_cancel_before_four_ranks_discards_partial_aggregate():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler.notify_external_load_completions(_notices()[:2])
    del scheduler.requests["request-0"]

    scheduler.notify_external_load_completions(_notices()[2:])

    assert scheduler._external_load_completion_ranks == {}
    connector.notify_external_load_completion.assert_not_called()


def test_ordinary_metadata_wins_then_late_rank_does_not_repopulate():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    connector.get_external_load_job_req_ids.return_value = {7: "request-0"}
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 4}
            ),
        )
    )
    connector.can_accept_external_load_completion.return_value = False

    scheduler.notify_external_load_completions([_notices()[0]])

    assert scheduler._external_load_completion_ranks == {}
    connector.notify_external_load_completion.assert_not_called()


def test_job_tombstone_does_not_swallow_later_load_same_request():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler._direct_external_load_completed_jobs = {7: ("request-0", 4)}
    connector.get_external_load_job_req_ids.side_effect = [
        {8: "request-0"},
        {},
    ]

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 4, 8: 4}
            ),
        )
    )

    assert scheduler.finished_recving_kv_req_ids == {"request-0"}
    assert scheduler._direct_external_load_completed_jobs == {}


def test_old_direct_finished_signal_does_not_finish_partial_new_load():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler._direct_external_load_completed_jobs = {7: ("request-0", 2)}
    connector.get_external_load_job_req_ids.side_effect = [
        {8: "request-0"},
        {8: "request-0"},
        {8: "request-0"},
        {},
        {9: "request-0"},
        {9: "request-0"},
        {9: "request-0"},
        {},
    ]

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 2, 8: 2}
            ),
        )
    )

    assert scheduler.finished_recving_kv_req_ids == set()
    assert scheduler._direct_external_load_completed_jobs == {}

    # Simulate the req-only KVOutputAggregator having consumed a mixed set of
    # old/new rank records above: it does not emit finished_recving again.
    # Job-aware completed_jobs must still make the later LOAD ready.
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={8: 2}
            ),
        )
    )

    assert scheduler.finished_recving_kv_req_ids == {"request-0"}

    scheduler.finished_recving_kv_req_ids.clear()
    # The req-only aggregator can remain offset after the mixed job7/job8
    # sequence and fire on only 2/4 records of job9. Job-aware pending state
    # must suppress that false edge.
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={9: 2}
            ),
        )
    )
    assert scheduler.finished_recving_kv_req_ids == set()

    # The real job9 completion remains authoritative even if the poisoned
    # req-only aggregator emits no second edge.
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={9: 2}
            ),
        )
    )
    assert scheduler.finished_recving_kv_req_ids == {"request-0"}


def test_reset_clears_partial_ranks_but_keeps_job_tombstone_for_late_normal():
    scheduler, connector, _ = _make_direct_completion_scheduler()
    scheduler.log_stats = False
    scheduler._external_load_completion_ranks = {
        8: ("request-1", {0}, 1.0)
    }
    scheduler._direct_external_load_completed_jobs = {7: ("request-0", 4)}
    connector.reset_cache.return_value = True

    assert scheduler.reset_connector_cache()

    assert scheduler._external_load_completion_ranks == {}
    assert scheduler._direct_external_load_completed_jobs == {
        7: ("request-0", 4)
    }


def test_cancel_fast_complete_reset_then_late_normal_frees_once():
    scheduler, connector, _ = _make_direct_completion_scheduler(
        RequestStatus.FINISHED_ABORTED
    )
    scheduler.log_stats = False
    connector.reset_cache.return_value = True

    scheduler.notify_external_load_completions(_notices())
    scheduler._external_load_completion_ranks = {
        8: ("request-1", {0}, 1.0)
    }
    assert scheduler.reset_connector_cache()
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_recving={"request-0"},
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 4}
            ),
        )
    )

    scheduler._free_blocks.assert_called_once()
    assert scheduler._external_load_completion_ranks == {}
    assert scheduler._direct_external_load_completed_jobs == {}


def test_external_lookup_defers_while_request_transfer_jobs_are_pending():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    request = SimpleNamespace(request_id="request-0")
    scheduler._req_status = {
        request.request_id: SimpleNamespace(transfer_jobs={3, 7})
    }
    scheduler._lookup = MagicMock()

    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    scheduler._lookup.assert_not_called()


def test_external_load_gpu_block_stats_are_exported_and_samples_drained():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace()
    scheduler._external_load_stage_latency_stats = defaultdict(list)
    scheduler._external_load_job_gpu_block_counts = [128, 512]
    scheduler._external_load_inflight_gpu_blocks = 512
    scheduler._external_load_peak_inflight_gpu_blocks = 640
    scheduler._external_load_max_job_gpu_blocks = 512
    scheduler._external_load_inflight_jobs = 1
    scheduler._external_load_peak_inflight_jobs = 2

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "external_load_gpu_block_job_count": 2,
        "external_load_gpu_block_job_sum": 640,
        "external_load_gpu_block_current_inflight_gpu_blocks": 512,
        "external_load_gpu_block_peak_inflight_gpu_blocks": 640,
        "external_load_gpu_block_max_job_gpu_blocks": 512,
        "external_load_gpu_block_current_inflight_jobs": 1,
        "external_load_gpu_block_peak_inflight_jobs": 2,
    }
    assert scheduler._external_load_job_gpu_block_counts == []


def test_external_load_gpu_extent_stats_are_exported_and_drained():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace()
    scheduler._external_load_stage_latency_stats = defaultdict(list)
    scheduler._external_load_gpu_extent_stats = defaultdict(
        int,
        {
            "requested_entries": 2,
            "allocator_selected_entries": 1,
            "allocator_fallback_entries": 1,
            "contiguous_entries": 1,
            "noncontiguous_entries": 1,
            "gpu_segments": 4,
            "gpu_segment_savings": 12,
        },
    )

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "external_load_gpu_extent_requested_entries": 2,
        "external_load_gpu_extent_allocator_selected_entries": 1,
        "external_load_gpu_extent_allocator_fallback_entries": 1,
        "external_load_gpu_extent_contiguous_entries": 1,
        "external_load_gpu_extent_noncontiguous_entries": 1,
        "external_load_gpu_extent_gpu_segments": 4,
        "external_load_gpu_extent_gpu_segment_savings": 12,
    }
    assert not scheduler._external_load_gpu_extent_stats


def test_external_load_host_latency_stats_are_exported():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace(
        take_external_load_stage_latency_stats=lambda: {
            "secondary_rank0_rpc": [0.2],
            "secondary_host_poll_delay": [0.05],
        }
    )
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "external_load_secondary_rank0_rpc_latency_count": 1,
        "external_load_secondary_rank0_rpc_latency_sum": 0.2,
        "external_load_secondary_host_poll_delay_latency_count": 1,
        "external_load_secondary_host_poll_delay_latency_sum": 0.05,
    }


def test_fused_store_stage_latency_stats_are_exported():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace(
        take_fused_store_stage_latency_stats=lambda: {
            "prepared_to_primary_ready": [0.1],
            "prepared_to_gpu_source_release_ready": [0.11],
            "primary_ready_to_primary_to_secondary_done": [0.3],
            "primary_to_secondary_done_to_secondary_ready": [0.02],
        }
    )
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "fused_store_prepared_to_primary_ready_latency_count": 1,
        "fused_store_prepared_to_primary_ready_latency_sum": 0.1,
        "fused_store_prepared_to_gpu_source_release_ready_latency_count": 1,
        "fused_store_prepared_to_gpu_source_release_ready_latency_sum": 0.11,
        "fused_store_primary_ready_to_primary_to_secondary_done_latency_count": 1,
        "fused_store_primary_ready_to_primary_to_secondary_done_latency_sum": 0.3,
        "fused_store_primary_to_secondary_done_to_secondary_ready_latency_count": 1,
        "fused_store_primary_to_secondary_done_to_secondary_ready_latency_sum": 0.02,
    }


def test_post_store_primary_ready_reuse_stats_are_exported():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace(
        take_post_store_primary_ready_reuse_stats=lambda: {
            "counters": {
                "cohort_keys": 5,
                "reused_keys": 2,
                "load_gpu_ready_key_events": 3,
                "reused_within_100ms_keys": 1,
                "reused_within_1s_keys": 2,
            },
            "latencies": {"to_first_load_gpu_ready": [0.05, 0.4]},
        }
    )
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "post_store_primary_ready_cohort_keys": 5,
        "post_store_primary_ready_reused_keys": 2,
        "post_store_primary_ready_load_gpu_ready_key_events": 3,
        "post_store_primary_ready_reused_within_100ms_keys": 1,
        "post_store_primary_ready_reused_within_1s_keys": 2,
        "post_store_primary_ready_to_first_load_gpu_ready_latency_count": 2,
        "post_store_primary_ready_to_first_load_gpu_ready_latency_sum": 0.45,
    }


def test_post_load_primary_published_cross_request_reuse_stats_are_exported():
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace(
        take_post_load_primary_published_cross_request_reuse_stats=lambda: {
            "counters": {
                "cohort_keys": 5,
                "origin_load_gpu_ready_key_events": 5,
                "reused_keys": 2,
                "reuse_load_gpu_ready_key_events": 3,
                "reused_within_1s_keys": 1,
                "reused_within_10s_keys": 2,
            },
            "latencies": {
                "load_primary_published_to_first_cross_request_"
                "load_gpu_ready": [0.4, 2.0]
            },
        }
    )
    scheduler._external_load_stage_latency_stats = defaultdict(list)

    stats = scheduler.get_kv_connector_stats()

    assert stats is not None
    assert stats.reduce() == {
        "post_load_primary_published_cross_request_cohort_keys": 5,
        "post_load_primary_published_cross_request_origin_load_gpu_ready_key_events": 5,
        "post_load_primary_published_cross_request_reused_keys": 2,
        "post_load_primary_published_cross_request_reuse_load_gpu_ready_key_events": 3,
        "post_load_primary_published_cross_request_reused_within_1s_keys": 1,
        "post_load_primary_published_cross_request_reused_within_10s_keys": 2,
        (
            "post_load_primary_published_to_first_cross_request_"
            "load_gpu_ready_latency_count"
        ): 2,
        (
            "post_load_primary_published_to_first_cross_request_"
            "load_gpu_ready_latency_sum"
        ): 2.4,
    }


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_preemption(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    free_block_queue = runner.scheduler.kv_cache_manager.block_pool.free_block_queue
    num_free_blocks_empty = free_block_queue.num_free_blocks

    # 2 blocks, store all, without flushing
    # blocks = [0, 1, 2], [3, 4, 5]
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0],
        complete_transfers=False,
    )

    # decode 2 more blocks - 1 gpu block, storing [6, 7, 8] (no flush)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (2 * offloaded_block_size - block_size),
        complete_transfers=False,
    )

    # simulate KV cache running out of space
    free_block_queue.num_free_blocks = 0

    # request should be preempted now
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
        expected_flushed=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        expected_stored=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    # restore KV cache space and reset GPU prefix cache
    free_block_queue.num_free_blocks = num_free_blocks_empty
    runner.scheduler.reset_prefix_cache()

    # request should now return from preemption
    # re-load [0, ..., 8] from the CPU and store [9, 10, 11]
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 3
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * block_size,
        expected_loaded=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )

    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(9, 10, 11),
    )

    # All stores completed before request_finished -> fence index empty.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_concurrent_lookups_of_the_same_prefix(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    # With sync scheduling, all-finished flush fires within this run.
    # With async scheduling, the finish is delayed so flush fires later.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
        expected_flushed=(0, 1, 2) if not async_scheduling else (),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # start a new request to load the same first block
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request did not trigger a load
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # complete transfers
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2),
    )

    # second request will use the GPU prefix cache
    assert transfer_jobs == list(runner.offloading_spec.handler.transfer_specs)

    # Fence index drained: stores completed before request_finished ran.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_abort_loading_requests(request_runner, async_scheduling: bool):
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # store 1 blocks
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
        expected_flushed=(0, 1, 2) if not async_scheduling else (),
    )

    # start a request to load the first block, but don't complete
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )

    # request triggered a load
    transfer_jobs = list(runner.offloading_spec.handler.transfer_specs)
    assert transfer_jobs

    # abort request
    req_id = str(runner.req_id)
    runner.scheduler.finish_requests((req_id,), RequestStatus.FINISHED_ABORTED)

    # verify request is not deleted
    assert req_id in runner.scheduler.requests

    # complete loading request
    runner.run(
        decoded_tokens=[],
        expected_loaded=(0, 1, 2),
        expected_flushed=(0, 1, 2),
    )

    # assert request is deleted
    assert req_id not in runner.scheduler.requests


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_two_groups_full_and_sliding_window(request_runner, async_scheduling: bool):
    block_size = 4
    num_gpu_blocks = 100
    # sliding_window=8 -> 2 offloaded blocks (block_size_factor=1)
    sliding_window = 8

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify group configs: group 0 = full attention, group 1 = sliding window
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    assert kv_group_configs[0].sliding_window_size_in_blocks is None
    assert kv_group_configs[1].sliding_window_size_in_blocks == 2

    # Blocks [0, 1, 2] miss
    runner.new_request(token_ids=[0] * block_size * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    # _touch called from get_num_new_matched_tokens (2 groups) and
    # _get_reqs_to_store (2 groups) → 4 touch calls total.
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 4
    assert len(touch_calls[0].args[0]) == 3
    assert len(touch_calls[1].args[0]) == 3
    assert len(touch_calls[2].args[0]) == 3
    assert len(touch_calls[3].args[0]) == 3

    # store 3 more block
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[0] * (block_size * 3 + 2),
        expected_stored=(0, 1, 2, 3, 4, 5),
    )

    # touch called from _get_reqs_to_store * 3 blocks, once for each group
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 6

    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    runner.scheduler.reset_prefix_cache()

    # full 3 blocks hit [0, 1, 2]
    runner.new_request(token_ids=[0] * (block_size * 3 + 1))
    runner.manager.lookup.return_value = True
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn): prefix lookup hits 3 → loads blocks 0,1,2
        # Group 1 (sliding window, window=2): only the last 2 blocks
        #   are within the window → loads blocks 1,2
        expected_loaded=((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    )

    # one touch in get_num_new_matched_tokens x 2 groups
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    # full attention group touched all 3 blocks
    assert len(touch_calls[0].args[0]) == 3
    # sliding window group touched just the last 2 blocks
    assert len(touch_calls[1].args[0]) == 2

    # 3 blocks are hit on GPU [0, 1, 2]
    # 1 block loaded [3,]
    runner.new_request(token_ids=[0] * (block_size * 4 + 1))
    runner.manager.lookup.return_value = True
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn): prefix lookup hits 3 → loads blocks 0,1,2
        # Group 1 (sliding window, window=2): only the last 2 blocks
        #   are within the window → loads blocks 1,2
        expected_loaded=((0, 3), (1, 3)),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_two_groups_different_block_sizes(request_runner, async_scheduling: bool):
    hash_block_size = 4
    num_gpu_blocks = 100

    # Group 0: block_size=12 (offloaded_block_size=12)
    # Group 1: block_size=16 (offloaded_block_size=16)
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=hash_block_size * 3,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            FullAttentionSpec(
                block_size=hash_block_size * 4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
    ]

    runner = request_runner(
        block_size=hash_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify group configs
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    assert kv_group_configs[0].gpu_block_size == 12
    assert kv_group_configs[0].offloaded_block_size == 12
    assert kv_group_configs[1].gpu_block_size == 16
    assert kv_group_configs[1].offloaded_block_size == 16

    # Prompt: 25 tokens, unaligned to both block sizes.
    # Group 0 blocks: [0, 1], ending_token_offset = 24
    # Group 1 blocks: [0,], ending_token_offset = 16
    runner.new_request(token_ids=[0] * 25)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    # _touch called from get_num_new_matched_tokens (2 groups) and
    # _get_reqs_to_store (2 groups) → 4 touch calls total.
    # Group 0 has 2 offload keys, group 1 has 1.
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 4
    assert len(touch_calls[0].args[0]) == 2
    assert len(touch_calls[1].args[0]) == 1
    assert len(touch_calls[2].args[0]) == 2
    assert len(touch_calls[3].args[0]) == 1

    # Get to 31 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 6, expected_stored=((0, 0), (0, 1), (1, 0)))

    # Get to 32 tokens
    # Group 0 blocks: [0, 1], ending_token_offset = 24
    # Group 1 blocks: [0, 1], ending_token_offset = 32
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: only group 1 has a new block to store
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 2
    assert len(touch_calls[1].args[0]) == 2

    # Get to 35 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 3, expected_stored=((1, 1),))

    # Get to 36 tokens
    # Group 0 blocks: [0, 1, 2], ending_token_offset = 36
    # Group 1 blocks: [0, 1], ending_token_offset = 32
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: only group 0 has a new block to store
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 3
    assert len(touch_calls[1].args[0]) == 2

    # Get to 47 tokens
    # No further blocks offloaded
    runner.run(decoded_tokens=[0] * 11, expected_stored=((0, 2),))

    # Get to 48 tokens
    # Group 0 blocks: [0, 1, 2, 3], ending_token_offset = 4
    # Group 1 blocks: [0, 1, 2], ending_token_offset = 48
    runner.run(decoded_tokens=[0])
    # _get_reqs_to_store touch: both groups have a new block, each with 1 key
    touch_calls = runner.manager.touch.call_args_list
    assert len(touch_calls) == 2
    assert len(touch_calls[0].args[0]) == 4
    assert len(touch_calls[1].args[0]) == 3

    runner.run(decoded_tokens=[0], expected_stored=((0, 3), (1, 2)))

    # Get to 96 tokens
    runner.run(
        decoded_tokens=[0] * 47 + [EOS_TOKEN_ID],
        expected_stored=((0, 4), (0, 5), (0, 6), (0, 7), (1, 3), (1, 4), (1, 5)),
    )

    runner.scheduler.reset_prefix_cache()

    # Request with 48 matching tokens
    # will match 48 tokens (4 block) from the first group
    # 48 tokens (3 block) from the second group
    # Total 48 tokens can be loaded
    runner.new_request(token_ids=[0] * 48)
    runner.manager.lookup.return_value = True
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[0],
        expected_loaded=((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # Request with 48+37 matching tokens
    # 48 tokens will be hit on GPU
    # extra 32 tokens will be loaded
    # extra tokens [0, 36] (blocks [4, 5, 6]) from the first group
    # extra tokens [0, 32] (block [3, 4]) from the second group
    runner.new_request(token_ids=[0] * (48 + 37))
    runner.manager.lookup.return_value = True
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[0],
        expected_loaded=((0, 4), (0, 5), (0, 6), (1, 3), (1, 4)),
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID])


# ---------------------------------------------------------------------------
# Unit tests for _maximal_prefix_lookup / _sliding_window_lookup
# ---------------------------------------------------------------------------


def _make_scheduler_with_lookup(
    lookup_results: dict[int, bool | None],
) -> OffloadingConnectorScheduler:
    """Create an OffloadingConnectorScheduler with a mocked manager.lookup."""
    manager = MagicMock(spec=OffloadingManager)
    manager.lookup.side_effect = lambda key, req_context: lookup_results.get(
        int(get_offload_block_hash(key).decode()), False
    )

    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = manager
    return scheduler


_EMPTY_REQ_CTX = ReqContext(req_id="")


class TestMaximalPrefixLookup:
    def test_all_hit(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 0

    def test_partial_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3]), _EMPTY_REQ_CTX) == 2

    def test_miss_then_hit(self):
        sched = _make_scheduler_with_lookup({2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) == 0

    def test_single_hit(self):
        sched = _make_scheduler_with_lookup({1: True})
        assert sched._maximal_prefix_lookup(to_keys([1]), _EMPTY_REQ_CTX) == 1

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._maximal_prefix_lookup([], _EMPTY_REQ_CTX) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: None, 2: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) is None

    def test_none_after_hit_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._maximal_prefix_lookup(to_keys([1, 2]), _EMPTY_REQ_CTX) is None

    def test_none_stops_at_miss(self):
        """None is treated as hit for iteration, but miss stops the scan."""
        sched = _make_scheduler_with_lookup({1: None, 2: False, 3: True})
        assert sched._maximal_prefix_lookup(to_keys([1, 2, 3]), _EMPTY_REQ_CTX) is None
        # lookup should have been called for blocks 1 and 2 (stops at miss)
        assert sched.manager.lookup.call_count == 2


class TestSlidingWindowLookup:
    def test_all_hit_exact_window(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2, _EMPTY_REQ_CTX) == 2

    def test_all_miss(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1, _EMPTY_REQ_CTX) == 0

    def test_window_at_end(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 2, _EMPTY_REQ_CTX) == 3

    def test_window_in_middle(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2, _EMPTY_REQ_CTX) == 3
        )

    def test_no_full_window_falls_back_to_prefix(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 3, _EMPTY_REQ_CTX) == 2

    def test_single_block_window(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True})
        assert sched._sliding_window_lookup(to_keys([1, 2, 3]), 1, _EMPTY_REQ_CTX) == 3

    def test_gap_resets_consecutive(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True})
        # [1, 2, 3, 0, 4] — gap at 0 resets, window of 2 found at [2,3]
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 0, 4]), 2, _EMPTY_REQ_CTX)
            == 3
        )

    def test_window_prefers_rightmost(self):
        sched = _make_scheduler_with_lookup({1: True, 2: True, 4: True, 5: True})
        # two valid windows: [1,2] at positions 0-1 and [4,5] at positions 3-4
        # scans right-to-left, finds [4,5] first
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4, 5]), 2, _EMPTY_REQ_CTX)
            == 5
        )

    def test_prefix_fallback_with_gap(self):
        sched = _make_scheduler_with_lookup({2: True, 3: True, 4: True, 5: True})
        # window of 4 not found contiguously (gap at 1)
        assert (
            sched._sliding_window_lookup(to_keys([2, 1, 3, 4, 5]), 4, _EMPTY_REQ_CTX)
            == 1
        )

    def test_empty(self):
        sched = _make_scheduler_with_lookup({})
        assert sched._sliding_window_lookup([], 1, _EMPTY_REQ_CTX) == 0

    def test_none_defers(self):
        sched = _make_scheduler_with_lookup({1: True, 2: None})
        assert sched._sliding_window_lookup(to_keys([1, 2]), 2, _EMPTY_REQ_CTX) is None

    def test_none_with_full_window_still_defers(self):
        """Even if a real window is found after a None, result is deferred."""
        # Scan right-to-left: 4(True), 3(None) resets, 2(True), 1(True) = window
        # but block 3 was None so defer_lookup is set
        sched = _make_scheduler_with_lookup({1: True, 2: True, 3: None, 4: True})
        assert (
            sched._sliding_window_lookup(to_keys([1, 2, 3, 4]), 2, _EMPTY_REQ_CTX)
            is None
        )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_request_level_policy_stores_all_blocks(request_runner, async_scheduling: bool):
    """With REQUEST_LEVEL policy, all blocks are stored — including prefix hits."""
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size_factor=block_size_factor,
        block_size=gpu_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
    )

    # Store 1 offloaded block (3 GPU blocks) via a normal request.
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
        expected_flushed=(0, 1, 2) if not async_scheduling else (),
    )

    # Reset GPU prefix cache so the next request must load from CPU.
    runner.scheduler.reset_prefix_cache()

    # Manager returns REQUEST_LEVEL for the next request.
    runner.manager.on_new_request.return_value = RequestOffloadingContext(
        policy=OffloadPolicy.REQUEST_LEVEL
    )

    # New request with 2 offloaded blocks; first matches what's in CPU.
    runner.new_request(token_ids=[0] * offloaded_block_size * 2)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # Load the first offloaded block from CPU.
    runner.run(decoded_tokens=[0], expected_loaded=(0, 1, 2))

    # Store must include ALL 6 GPU blocks (both the loaded prefix and
    # the newly computed block), not just the 3 new ones.
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3, 4, 5))

    # All stores completed before request_finished -> fence index empty.
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


# ---------------------------------------------------------------------------
# Tests for the per-job-store-completion design and fence invariants.
# ---------------------------------------------------------------------------


def _make_source_fence_scheduler(
    *, request_finished: bool = False, keep_scheduler_active: bool = False
):
    scheduler = object.__new__(OffloadingConnectorScheduler)
    request = MagicMock()
    request.is_finished.return_value = request_finished
    req_status = SimpleNamespace(
        req=request,
        req_context=MagicMock(),
        transfer_jobs={7},
    )
    scheduler._req_status = {"request-0": req_status}
    if keep_scheduler_active:
        active_request = MagicMock()
        active_request.is_finished.return_value = False
        scheduler._req_status["active"] = SimpleNamespace(
            req=active_request,
            req_context=MagicMock(),
            transfer_jobs=set(),
        )
    scheduler._jobs = {
        7: TransferJobStatus(
            req_id="request-0",
            pending_count=4,
            keys=set(to_keys([1])),
            is_store=True,
            non_sliding_window_block_ids=[10, 11, 12],
            sliding_window_block_ids=[20],
        )
    }
    indexed_block_ids = (10, 11, 12, 20) if request_finished else (20,)
    scheduler._block_id_to_pending_jobs = {
        block_id: {7} for block_id in indexed_block_ids
    }
    scheduler._current_batch_load_jobs = {}
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._current_batch_store_source_blocks_to_flush = defaultdict(set)
    scheduler._current_batch_store_source_jobs_to_retire = set()
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._stale_job_threshold = 0
    scheduler._update_req_states = MagicMock()
    scheduler._build_store_jobs = MagicMock(return_value={})
    scheduler.manager = MagicMock()
    return scheduler


def test_reallocated_store_source_uses_exact_block_fence():
    scheduler = _make_source_fence_scheduler(
        request_finished=True, keep_scheduler_active=True
    )
    scheduler._current_batch_allocated_block_ids = {11}

    metadata = scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=[])
    )

    assert metadata.jobs_to_flush == set()
    assert metadata.store_source_blocks_to_flush == {7: (11,)}
    assert scheduler._jobs[7].non_sliding_window_block_ids == [10, 12]
    assert 11 not in scheduler._block_id_to_pending_jobs


def test_preempted_store_uses_source_fence_but_all_finished_uses_whole_job():
    scheduler = _make_source_fence_scheduler()

    metadata = scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=["request-0"])
    )

    assert metadata.jobs_to_flush == set()
    assert metadata.store_source_blocks_to_flush == {
        7: (10, 11, 12, 20)
    }

    metadata = scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=["request-0"])
    )
    assert metadata.jobs_to_flush == set()
    assert metadata.store_source_blocks_to_flush == {}

    scheduler = _make_source_fence_scheduler(request_finished=True)
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=["request-0"])
    )

    assert metadata.jobs_to_flush == {7}
    assert metadata.store_source_blocks_to_flush == {}


def test_store_source_state_retires_only_after_all_tp_workers_complete():
    scheduler = _make_source_fence_scheduler(
        request_finished=True, keep_scheduler_active=True
    )

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 3}
            )
        )
    )
    assert scheduler._current_batch_store_source_jobs_to_retire == set()

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={7: 1}
            )
        )
    )
    assert scheduler._current_batch_store_source_jobs_to_retire == {7}
    scheduler.manager.complete_store.assert_called_once()


def test_loads_do_not_populate_fence_index(request_runner):
    """Loads don't populate _block_id_to_pending_jobs (protected by
    delay_free_blocks while in flight)."""
    runner = request_runner(
        block_size_factor=3,
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=False,
    )
    runner.new_request(token_ids=[0] * 12)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.run(decoded_tokens=[], complete_transfers=False)
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


def test_fence_at_update_state_after_alloc(request_runner):
    """A load reusing a finished request's pending-store block triggers
    a flush via update_state_after_alloc's fence.

    num_gpu_blocks=2 forces the BlockPool to give req2 the same block
    req1 just freed.
    """
    runner = request_runner(
        block_size_factor=1,
        block_size=4,
        num_gpu_blocks=2,
        async_scheduling=False,
    )

    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        expected_stored=(0,),
        expected_flushed=(0,),
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * 4)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[],
        complete_transfers=False,
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


def test_fence_at_build_store_jobs(request_runner):
    """A new prefill (no load -> update_state_after_alloc returns early)
    reusing a finished request's pending-store block is flushed by
    _build_store_jobs's fence."""
    runner = request_runner(
        block_size_factor=1,
        block_size=4,
        num_gpu_blocks=2,
        async_scheduling=False,
    )

    runner.new_request(token_ids=[0] * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        expected_stored=(0,),
        expected_flushed=(0,),
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}

    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[1] * 4)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 0
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
    )
    assert runner.connector_scheduler._block_id_to_pending_jobs == {}


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_complete_store_called_per_job(request_runner, async_scheduling: bool):
    """complete_store fires per-job, not deferred to request finish.
    Each call carries only that store's keys."""
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor
    runner = request_runner(
        block_size_factor=block_size_factor,
        block_size=gpu_block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
    )
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )

    # First store: fires when block 0 is fully populated.
    runner.run(decoded_tokens=[0, 0], expected_stored=(0, 1, 2))
    assert runner.manager.complete_store.call_count == 1
    first_call_keys = set(runner.manager.complete_store.call_args.args[0])
    assert len(first_call_keys) == 1
    runner.manager.complete_store.reset_mock()

    # Second store: fires when block 1 is fully populated, with different keys.
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size + 1),
        expected_stored=(3, 4, 5),
    )
    assert runner.manager.complete_store.call_count == 1
    second_call_keys = set(runner.manager.complete_store.call_args.args[0])
    assert first_call_keys != second_call_keys
    runner.manager.complete_store.reset_mock()

    # Finish: no store pending -> no further call.
    runner.run(decoded_tokens=[EOS_TOKEN_ID])
    assert runner.manager.complete_store.call_count == 0


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_max_offload_tokens_validation(request_runner, async_scheduling: bool):
    """Validates max_offload_tokens: type coercion, boundary values, and capping.

    Setup: 3 offloaded blocks × 3 GPU blocks each = 9 GPU block offsets (0–8).
    """
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor  # 12
    num_gpu_blocks = 100
    all_offsets = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def make_runner():
        return request_runner(
            block_size=gpu_block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
            block_size_factor=block_size_factor,
        )

    def setup(r, max_offload_tokens):
        r.new_request(
            token_ids=[0] * offloaded_block_size * 3,
            kv_transfer_params={"max_offload_tokens": max_offload_tokens},
        )
        r.manager.prepare_store.side_effect = (
            lambda keys, req_context: generate_store_output(keys)
        )

    # With sync scheduling, the connector flushes completed stores when the
    # request finishes; async scheduling defers the flush to the next step.
    flushed_all = all_offsets if not async_scheduling else ()
    flushed_two = (0, 1, 2, 3, 4, 5) if not async_scheduling else ()

    # None -> no cap, all 9 offsets stored
    r = make_runner()
    setup(r, None)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # str -> warn and fall back to no cap
    r = make_runner()
    setup(r, "24")
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # float -> warn and fall back to no cap
    r = make_runner()
    setup(r, 24.5)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # negative -> warn and fall back to no cap
    r = make_runner()
    setup(r, -1)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # bool -> rejected (type(True) is bool, not int), falls back to no cap
    r = make_runner()
    setup(r, True)
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=all_offsets,
        expected_flushed=flushed_all,
    )

    # zero -> valid, no blocks offloaded
    r = make_runner()
    setup(r, 0)
    r.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=())

    # positive int cap -> limits offload to first 2 offloaded blocks (offsets 0–5)
    r = make_runner()
    setup(r, 24)  # 24 tokens = 2 offloaded blocks × 12 tokens each
    r.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2, 3, 4, 5),
        expected_flushed=flushed_two,
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_offload_prompt_only(request_runner, async_scheduling: bool):
    """offload_prompt_only=True offloads prompt blocks but never decode blocks.

    Setup: a 2-offloaded-block prompt followed by enough decode tokens to fill
    4 more offloaded blocks. The flag clamps the offloadable token count to the
    prompt length, so only the prompt's blocks (GPU offsets 0-5) are ever
    eligible for store; the decode blocks (offsets >= 6) are skipped.

    The request is intentionally not terminated (no EOS): a store is only
    flushed when a request finishes (or is preempted), so without a finish
    there is nothing to flush and the assertion stays free of flush-timing
    subtleties. The decode steps are still enough for the prompt store to
    complete and show up in expected_stored.
    """
    gpu_block_size = 4
    block_size_factor = 3
    offloaded_block_size = gpu_block_size * block_size_factor  # 12
    num_prompt_blocks = 2
    num_decode_blocks = 4
    prompt_offsets = (0, 1, 2, 3, 4, 5)

    runner = request_runner(
        block_size=gpu_block_size,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
        extra_config_overrides={"offload_prompt_only": True},
    )

    runner.manager.prepare_store.side_effect = (
        lambda keys, req_context: generate_store_output(keys)
    )

    runner.new_request(token_ids=[0] * offloaded_block_size * num_prompt_blocks)
    runner.run(
        decoded_tokens=[0] * (offloaded_block_size * num_decode_blocks),
        expected_stored=prompt_offsets,
    )

    # Timing-independent guard: only the prompt's blocks were ever offered for
    # store. If decode blocks leaked through, more keys would appear here.
    offered_keys = {
        key
        for call in runner.manager.prepare_store.call_args_list
        for key in call.args[0]
    }
    assert len(offered_keys) == num_prompt_blocks


def test_flush_all_jobs_when_no_requests_remain(request_runner):
    """When all tracked requests are finished, build_connector_meta flushes
    all pending jobs since there will be no future step to complete them."""
    block_size = 4
    block_size_factor = 1
    offloaded_block_size = block_size * block_size_factor

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=False,
        block_size_factor=block_size_factor,
    )

    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        complete_transfers=False,
        expected_stored=(0,),
        expected_flushed=(0,),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_reset_cache(request_runner, async_scheduling: bool):
    """reset_cache flushes in-flight loads, calls manager.reset_cache(), resets
    next_stored_block_idx for active requests and clears job tracking."""
    block_size = 4
    block_size_factor = 3
    offloaded_block_size = block_size * block_size_factor
    num_gpu_blocks = 100

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        block_size_factor=block_size_factor,
    )

    # Store 1 offloaded block (3 GPU blocks) to CPU.
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(0, 1, 2),
        expected_flushed=(0, 1, 2) if not async_scheduling else (),
    )

    # Reset GPU prefix cache then start a request that loads from CPU.
    # Leave the load in-flight so that reset_cache must flush it.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * offloaded_block_size)
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[], complete_transfers=False)

    # Capture in-flight load job IDs before reset.
    load_job_ids = {
        jid
        for jid, status in runner.connector_scheduler._jobs.items()
        if not status.is_store
    }
    assert load_job_ids, "expected in-flight load jobs before reset"

    # Record job counter to verify the reset counter is set correctly.
    job_counter_before_reset = runner.connector_scheduler._job_counter

    # After update_state_after_alloc, next_stored_block_idx is advanced to
    # skip the loaded prefix; reset_cache must bring it back to 0.
    for req_status in runner.connector_scheduler._req_status.values():
        for group_state in req_status.group_states:
            assert group_state.next_stored_block_idx > 0

    # Reset the cache
    runner.connector_scheduler.reset_cache()

    # manager.reset_cache() must be called exactly once.
    runner.manager.reset_cache.assert_called_once()

    # In-flight load jobs must be queued for flushing to prevent CUDA stream
    # races between old loads and new post-reset stores.
    assert load_job_ids <= runner.connector_scheduler._current_batch_jobs_to_flush

    # All internal job tracking must be cleared.
    assert not runner.connector_scheduler._jobs
    assert not runner.connector_scheduler._block_id_to_pending_jobs
    if runner.connector_scheduler._blocks_being_loaded is not None:
        assert not runner.connector_scheduler._blocks_being_loaded

    # Job reset counter must equal the job counter so that completions for
    # pre-reset jobs arriving from workers are silently discarded.
    assert runner.connector_scheduler._stale_job_threshold == job_counter_before_reset

    # next_stored_block_idx must be reset to 0 for every active request so
    # that post-reset stores restart from block 0.
    for req_status in runner.connector_scheduler._req_status.values():
        for group_state in req_status.group_states:
            assert group_state.next_stored_block_idx == 0


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_swa_alignment_skip(request_runner, async_scheduling: bool):
    """SWA blocks unreachable by the load path are skipped during store.

    Simulates a DeepSeek V4-like hybrid architecture where SWA groups have
    much smaller block sizes than the full-attention (MLA) group, causing
    most SWA blocks to be unreachable by the alignment-based load path.

    Setup:
      - Group 0: full attention (MLA-like), block_size=16
      - Group 1: SWA, block_size=4, sliding_window=8

    alignment_block_count = 16 / 4 = 4 SWA blocks per alignment segment.
    sliding_window_size_in_blocks = ceil(8 / 4) = 2.
    Within each segment of 4 SWA blocks, only the trailing 2 are stored.

    With 32 tokens (2 full-attn blocks, 8 SWA blocks):
      - Group 0 stores: blocks 0, 1  (all full-attn blocks)
      - Group 1 stores: blocks 2, 3, 6, 7  (skip 0,1,4,5)

    For real DeepSeek V4 (100K tokens), this reduces SWA stores by ~78%.
    """
    full_attn_block_size = 16
    swa_block_size = 4
    sliding_window = 8
    num_gpu_blocks = 200

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=full_attn_block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=swa_block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=swa_block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Verify config: alignment_block_count computed correctly
    kv_group_configs = runner.connector_scheduler.config.kv_group_configs
    assert len(kv_group_configs) == 2
    # Group 0: full attention -> no alignment skip
    assert kv_group_configs[0].alignment_block_count is None
    assert kv_group_configs[0].sliding_window_size_in_blocks is None
    assert kv_group_configs[0].offloaded_block_size == full_attn_block_size
    # Group 1: SWA -> alignment_block_count = 16/4 = 4, tail = 2
    assert kv_group_configs[1].alignment_block_count == 4
    assert kv_group_configs[1].sliding_window_size_in_blocks == 2
    assert kv_group_configs[1].offloaded_block_size == swa_block_size

    # Send 32 tokens = 2 full-attn blocks (block_size=16) = 8 SWA blocks
    # (block_size=4). Decode 1 token to kick off processing (stores are
    # deferred to next step).
    num_tokens = 32
    runner.new_request(token_ids=[0] * num_tokens)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])

    # Decode 1 more token to complete the deferred stores from above.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0 (full attn, block_size=16): 2 offloaded blocks
        #   -> GPU blocks (0, 0) and (0, 1)
        # Group 1 (SWA, block_size=4): 8 offloaded blocks, skip first 2
        #   per segment of 4:
        #   Segment 0 (blocks 0-3): skip 0,1 -> store (1, 2), (1, 3)
        #   Segment 1 (blocks 4-7): skip 4,5 -> store (1, 6), (1, 7)
        expected_stored=(
            (0, 0),
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
        ),
    )

    # Verify that loads still work correctly for the stored SWA blocks.
    runner.scheduler.reset_prefix_cache()
    runner.new_request(token_ids=[0] * num_tokens + [1])
    runner.manager.lookup.return_value = True
    runner.connector_scheduler._maximal_prefix_lookup = lambda key, req_context: 2
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        # Group 0: full prefix lookup hits 2 offloaded blocks
        #   -> loads GPU blocks (0, 0), (0, 1)
        # Group 1: sliding window lookup finds trailing 2 from last segment
        #   (blocks 6, 7 which were stored)
        #   -> loads GPU blocks (1, 6), (1, 7)
        expected_loaded=(
            (0, 0),
            (0, 1),
            (1, 6),
            (1, 7),
        ),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_stale_sliding_window_block_after_prepare_store_failure(
    request_runner, async_scheduling: bool
):
    """Regression test: when prepare_store fails (returns None), offloading is
    delayed. Meanwhile, sliding window blocks get freed and reallocated to the
    same request. On retry, the stale block_id must be detected and skipped.

    Without the fix, the stale block_id would either:
    - Cause a KeyError in _remove_pending_job (duplicate in
      _block_id_to_pending_jobs)
    - Silently offload wrong data under a wrong key
    """
    block_size = 4
    # sliding_window = 8 -> window of 2 blocks
    sliding_window = 8
    # Use a tight GPU block budget so freed sliding window blocks are
    # immediately reused by the same request's new allocations.
    num_gpu_blocks = 4

    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]

    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
    )

    # Request with 3 blocks of prompt. Window = 2 blocks, so block 0 is
    # outside the window but won't be freed until the next allocate_slots.
    runner.new_request(token_ids=[0] * block_size * 3)

    # First step: prepare_store FAILS -> offloading delayed.
    # next_stored_block_idx stays at 0, block_ids[0] still holds the
    # original block_id for position 0.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.assert_called()

    # Second step: decode more tokens -> block 3 allocated.
    # allocate_slots calls remove_skipped_blocks which frees block 0
    # (it's now outside the sliding window). With num_gpu_blocks=4,
    # the freed block is immediately reused for the new allocation.
    # prepare_store still fails so offloading is still delayed.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: None
    runner.run(decoded_tokens=[0] * block_size)

    # Now prepare_store succeeds.
    # Without the fix, the request would try to offload the stale block_id
    # at position 0 (now reused at position 3), causing a duplicate in
    # sliding_window_block_ids and eventually a KeyError.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    # block_ids=[0, ?, 3, 1]: positions 0 and 1 are zeroed (stale blocks that
    # were freed by the sliding window and reallocated). Only blocks at
    # positions 2 and 3 (request offsets 2, 3) are stored.
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(2, 3),
        expected_flushed=(2, 3) if not async_scheduling else (),
    )
