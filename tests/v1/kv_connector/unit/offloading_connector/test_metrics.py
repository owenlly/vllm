# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    EXTERNAL_LOAD_GPU_BLOCK_BUCKETS,
    KV_OFFLOAD_SIZE_BUCKETS_BYTES,
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def test_build_kv_connector_stats_with_none():
    """Test that build_kv_connector_stats returns empty stats when given None."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_kv_offload_size_buckets_cover_full_eight_gib_pool():
    """The job-size CDF has a finite bound for every supported DPU job."""
    assert tuple(sorted(set(KV_OFFLOAD_SIZE_BUCKETS_BYTES))) == (
        KV_OFFLOAD_SIZE_BUCKETS_BYTES
    )
    assert KV_OFFLOAD_SIZE_BUCKETS_BYTES[-1] == 8 * 1024**3
    assert KV_OFFLOAD_SIZE_BUCKETS_BYTES[:10] == (
        1_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        40_000_000,
        60_000_000,
        80_000_000,
        100_000_000,
        150_000_000,
        200_000_000,
    )


def test_external_load_gpu_block_buckets_cover_large_jobs():
    assert tuple(sorted(set(EXTERNAL_LOAD_GPU_BLOCK_BUCKETS))) == (
        EXTERNAL_LOAD_GPU_BLOCK_BUCKETS
    )
    assert EXTERNAL_LOAD_GPU_BLOCK_BUCKETS[0] == 1
    assert EXTERNAL_LOAD_GPU_BLOCK_BUCKETS[-1] == 16384


def test_aggregate_external_load_gpu_block_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_external_load_gpu_block_stats(
        {
            "job_gpu_blocks": [8, 128],
            "current_inflight_gpu_blocks": 128,
            "peak_inflight_gpu_blocks": 256,
            "max_job_gpu_blocks": 128,
            "current_inflight_jobs": 1,
            "peak_inflight_jobs": 3,
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_external_load_gpu_block_stats(
        {
            "job_gpu_blocks": [64, 512],
            "current_inflight_gpu_blocks": 0,
            "peak_inflight_gpu_blocks": 640,
            "max_job_gpu_blocks": 512,
            "current_inflight_jobs": 0,
            "peak_inflight_jobs": 4,
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced == {
        "external_load_gpu_block_job_count": 4,
        "external_load_gpu_block_job_sum": 712,
        "external_load_gpu_block_current_inflight_gpu_blocks": 0,
        "external_load_gpu_block_peak_inflight_gpu_blocks": 640,
        "external_load_gpu_block_max_job_gpu_blocks": 512,
        "external_load_gpu_block_current_inflight_jobs": 0,
        "external_load_gpu_block_peak_inflight_jobs": 4,
    }


@pytest.mark.parametrize(
    "bad_stats",
    [
        {"job_gpu_blocks": [-1]},
        {"job_gpu_blocks": [1.5]},
        {"job_gpu_blocks": "8"},
        {"current_inflight_gpu_blocks": -1},
        {"unknown": 1},
    ],
)
def test_external_load_gpu_block_stats_reject_invalid_values(bad_stats):
    with pytest.raises(ValueError):
        OffloadingConnectorStats().record_external_load_gpu_block_stats(
            bad_stats
        )


def test_aggregate_external_load_gpu_extent_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_external_load_gpu_extent_stats(
        {
            "requested_entries": 2,
            "allocator_selected_entries": 1,
            "allocator_fallback_entries": 1,
            "contiguous_entries": 1,
            "noncontiguous_entries": 1,
            "gpu_segments": 3,
            "gpu_segment_savings": 13,
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_external_load_gpu_extent_stats(
        {
            "requested_entries": 3,
            "allocator_selected_entries": 2,
            "allocator_fallback_entries": 1,
            "contiguous_entries": 2,
            "noncontiguous_entries": 1,
            "gpu_segments": 5,
            "gpu_segment_savings": 19,
        }
    )

    assert stats1.aggregate(stats2).reduce() == {
        "external_load_gpu_extent_requested_entries": 5,
        "external_load_gpu_extent_allocator_selected_entries": 3,
        "external_load_gpu_extent_allocator_fallback_entries": 2,
        "external_load_gpu_extent_contiguous_entries": 3,
        "external_load_gpu_extent_noncontiguous_entries": 2,
        "external_load_gpu_extent_gpu_segments": 8,
        "external_load_gpu_extent_gpu_segment_savings": 32,
    }


@pytest.mark.parametrize(
    "bad_stats",
    [
        {"requested_entries": -1},
        {"gpu_segments": 1.5},
        {"unknown": 1},
    ],
)
def test_external_load_gpu_extent_stats_reject_invalid_values(bad_stats):
    with pytest.raises(ValueError):
        OffloadingConnectorStats().record_external_load_gpu_extent_stats(
            bad_stats
        )


def test_build_kv_connector_stats_with_empty_dict():
    """Test that build_kv_connector_stats returns empty stats with empty dict."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """Test that OffloadingConnector stats are properly reconstructed with
    correct data."""
    serialized_data = {
        "CPU_to_GPU": [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ],
        "GPU_to_CPU": [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
        ],
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    offload_connector_stats = stats
    assert isinstance(offload_connector_stats, OffloadingConnectorStats)
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
    ]


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    stats1 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
            ],
        }
    )

    stats2 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    offload_connector_stats = result
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
        {"op_size": 3, "op_time": 0.2},
        {"op_size": 7, "op_time": 0.9},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
        {"op_size": 16, "op_time": 2},
    ]


def test_aggregate_rdma_transport_stats():
    """RDMA counters are summed by logical STORE/LOAD across TP workers."""
    stats1 = OffloadingConnectorStats()
    stats1.record_rdma_transport_stats(
        "store",
        {
            "logical_requests": 10,
            "wr_posted": 4,
            "sge_posted": 20,
            "multi_sge_wr": 4,
            "wr_split": 0,
        },
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_rdma_transport_stats(
        "store",
        {
            "logical_requests": 6,
            "wr_posted": 3,
            "sge_posted": 12,
            "multi_sge_wr": 3,
            "wr_split": 1,
        },
    )
    stats2.record_rdma_transport_stats(
        "load",
        {
            "logical_requests": 2,
            "wr_posted": 2,
            "sge_posted": 8,
            "multi_sge_wr": 2,
            "wr_split": 0,
        },
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["rdma_store_logical_requests"] == 16
    assert reduced["rdma_store_wr_posted"] == 7
    assert reduced["rdma_store_sge_posted"] == 32
    assert reduced["rdma_store_multi_sge_wr"] == 7
    assert reduced["rdma_store_wr_split"] == 1
    assert reduced["rdma_load_logical_requests"] == 2
    assert reduced["rdma_load_wr_posted"] == 2
    assert reduced["rdma_load_sge_posted"] == 8
    assert reduced["rdma_load_multi_sge_wr"] == 2
    assert reduced["rdma_load_wr_split"] == 0


def test_aggregate_secondary_tier_stats():
    """Test that secondary tier counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_secondary_tier_stats(
        {
            "store": {
                "candidate_blocks": 10,
                "skipped_present_blocks": 4,
                "submitted_jobs": 2,
                "submitted_blocks": 6,
                "succeeded_jobs": 1,
                "succeeded_blocks": 3,
            },
            "promotion": {
                "submitted_jobs": 1,
                "submitted_blocks": 2,
            },
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_secondary_tier_stats(
        {
            "store": {
                "succeeded_jobs": 1,
                "succeeded_blocks": 3,
                "failed_jobs": 1,
                "failed_blocks": 2,
            },
            "promotion": {
                "succeeded_jobs": 1,
                "succeeded_blocks": 2,
            },
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["secondary_tier_store_submitted_jobs"] == 2
    assert reduced["secondary_tier_store_candidate_blocks"] == 10
    assert reduced["secondary_tier_store_skipped_present_blocks"] == 4
    assert reduced["secondary_tier_store_submitted_blocks"] == 6
    assert reduced["secondary_tier_store_succeeded_jobs"] == 2
    assert reduced["secondary_tier_store_succeeded_blocks"] == 6
    assert reduced["secondary_tier_store_failed_jobs"] == 1
    assert reduced["secondary_tier_store_failed_blocks"] == 2
    assert reduced["secondary_tier_promotion_submitted_jobs"] == 1
    assert reduced["secondary_tier_promotion_submitted_blocks"] == 2
    assert reduced["secondary_tier_promotion_succeeded_jobs"] == 1
    assert reduced["secondary_tier_promotion_succeeded_blocks"] == 2


def test_aggregate_secondary_tier_job_latency_stats():
    """Test that secondary tier job latencies are preserved for histograms."""
    stats1 = OffloadingConnectorStats()
    stats1.record_secondary_tier_job_latency_stats(
        {
            "store": {
                "succeeded": [0.01, 0.02],
            },
            "promotion": {
                "succeeded": [0.03],
            },
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_secondary_tier_job_latency_stats(
        {
            "store": {
                "succeeded": [0.04],
                "failed": [0.05],
            },
            "promotion": {
                "failed": [0.06],
            },
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["secondary_tier_store_succeeded_latency_count"] == 3
    assert (
        abs(reduced["secondary_tier_store_succeeded_latency_sum"] - 0.07)
        < 1e-12
    )
    assert reduced["secondary_tier_store_failed_latency_count"] == 1
    assert reduced["secondary_tier_store_failed_latency_sum"] == 0.05
    assert reduced["secondary_tier_promotion_succeeded_latency_count"] == 1
    assert reduced["secondary_tier_promotion_succeeded_latency_sum"] == 0.03
    assert reduced["secondary_tier_promotion_failed_latency_count"] == 1
    assert reduced["secondary_tier_promotion_failed_latency_sum"] == 0.06


def test_aggregate_external_load_stage_latency_stats():
    """Request-visible LOAD stage samples are preserved across workers."""
    stats1 = OffloadingConnectorStats()
    stats1.record_external_load_stage_latency_stats(
        {
            "secondary_submit_to_primary_published": [0.3],
            "scheduler_gap": [0.02],
            "scheduler_job_created_to_gpu_ready": [0.04],
            "secondary_submit_to_gpu_ready": [0.36],
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_external_load_stage_latency_stats(
        {
            "secondary_submit_to_primary_published": [0.5],
            "scheduler_gap": [0.03],
            "scheduler_job_created_to_gpu_ready": [0.07],
            "secondary_submit_to_gpu_ready": [0.6],
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert (
        reduced[
            "external_load_secondary_submit_to_primary_published_latency_count"
        ]
        == 2
    )
    assert (
        reduced[
            "external_load_secondary_submit_to_primary_published_latency_sum"
        ]
        == 0.8
    )
    assert reduced["external_load_scheduler_gap_latency_count"] == 2
    assert reduced["external_load_scheduler_gap_latency_sum"] == 0.05
    assert (
        reduced[
            "external_load_scheduler_job_created_to_gpu_ready_latency_count"
        ]
        == 2
    )
    assert (
        abs(
            reduced[
                "external_load_scheduler_job_created_to_gpu_ready_latency_sum"
            ]
            - 0.11
        )
        < 1e-12
    )
    assert reduced["external_load_secondary_submit_to_gpu_ready_latency_count"] == 2
    assert (
        abs(
            reduced["external_load_secondary_submit_to_gpu_ready_latency_sum"]
            - 0.96
        )
        < 1e-12
    )


def test_aggregate_external_load_completion_notify_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_external_load_completion_notify_stats(
        {"emitted": 2, "fast_completions": 1},
        {"rank_aggregate": [0.01]},
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_external_load_completion_notify_stats(
        {"emitted": 3, "dropped": 1},
        {"rank_aggregate": [0.02]},
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["external_load_completion_notify_emitted"] == 5
    assert reduced["external_load_completion_notify_dropped"] == 1
    assert reduced["external_load_completion_notify_fast_completions"] == 1
    assert reduced[
        "external_load_completion_notify_rank_aggregate_latency_count"
    ] == 2
    assert reduced[
        "external_load_completion_notify_rank_aggregate_latency_sum"
    ] == pytest.approx(0.03)


def test_aggregate_dpu_lookup_stage_latency_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_dpu_lookup_stage_latency_stats(
        {
            "request_first_lookup_to_dispatch": [0.01],
            "rank0_bf3_service": [0.00002],
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_dpu_lookup_stage_latency_stats(
        {
            "request_first_lookup_to_dispatch": [0.02],
            "rank0_bf3_service": [0.00003],
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced[
        "dpu_lookup_request_first_lookup_to_dispatch_latency_count"
    ] == 2
    assert reduced[
        "dpu_lookup_request_first_lookup_to_dispatch_latency_sum"
    ] == pytest.approx(0.03)
    assert reduced["dpu_lookup_rank0_bf3_service_latency_count"] == 2
    assert reduced["dpu_lookup_rank0_bf3_service_latency_sum"] == pytest.approx(
        0.00005
    )


def test_aggregate_fused_store_stage_latency_stats():
    """Fused STORE stage samples are preserved across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_fused_store_stage_latency_stats(
        {
            "prepared_to_primary_ready": [0.1],
            "prepared_to_gpu_source_release_ready": [0.11],
            "primary_ready_to_primary_to_secondary_done": [0.3],
            "primary_to_secondary_done_to_secondary_ready": [0.02],
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_fused_store_stage_latency_stats(
        {
            "prepared_to_primary_ready": [0.2],
            "prepared_to_gpu_source_release_ready": [0.21],
            "primary_ready_to_primary_to_secondary_done": [0.4],
            "primary_to_secondary_done_to_secondary_ready": [0.03],
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert (
        reduced[
            "fused_store_prepared_to_primary_ready_latency_count"
        ]
        == 2
    )
    assert (
        abs(
            reduced[
                "fused_store_prepared_to_primary_ready_latency_sum"
            ]
            - 0.3
        )
        < 1e-12
    )
    assert (
        reduced[
            "fused_store_prepared_to_gpu_source_release_ready_latency_count"
        ]
        == 2
    )
    assert (
        abs(
            reduced[
                "fused_store_prepared_to_gpu_source_release_ready_latency_sum"
            ]
            - 0.32
        )
        < 1e-12
    )
    assert (
        reduced[
            "fused_store_primary_ready_to_primary_to_secondary_done_latency_count"
        ]
        == 2
    )
    assert (
        abs(
            reduced[
                "fused_store_primary_ready_to_primary_to_secondary_done_latency_sum"
            ]
            - 0.7
        )
        < 1e-12
    )
    assert (
        reduced[
            "fused_store_primary_to_secondary_done_to_secondary_ready_latency_count"
        ]
        == 2
    )
    assert (
        abs(
            reduced[
                "fused_store_primary_to_secondary_done_to_secondary_ready_latency_sum"
            ]
            - 0.05
        )
        < 1e-12
    )


def test_aggregate_fused_store_dpu_dedup_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_fused_store_dedup_stats(
        {
            "candidate_entries": 10,
            "present_skipped_entries": 4,
            "writing_skipped_entries": 2,
            "transferred_entries": 4,
            "work_requests": 3,
            "bytes": 48 << 20,
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_fused_store_dedup_stats(
        {
            "candidate_entries": 5,
            "present_skipped_entries": 1,
            "writing_skipped_entries": 1,
            "transferred_entries": 3,
            "work_requests": 2,
            "bytes": 36 << 20,
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["fused_store_dedup_candidate_entries"] == 15
    assert reduced["fused_store_dedup_present_skipped_entries"] == 5
    assert reduced["fused_store_dedup_writing_skipped_entries"] == 3
    assert reduced["fused_store_dedup_transferred_entries"] == 7
    assert reduced["fused_store_dedup_work_requests"] == 5
    assert reduced["fused_store_dedup_bytes"] == 84 << 20


def test_aggregate_fused_resident_stats():
    stats1 = OffloadingConnectorStats()
    stats1.record_fused_resident_stats(
        "load",
        {
            "resident_hit_entries": 3,
            "admitted_entries": 4,
            "reject_frequency_entries": 2,
            "urgent_resident_hit_entries": 2,
            "urgent_admitted_entries": 3,
            "urgent_candidate_evictions": 1,
            "resident_bytes": 84,
            "staging_bytes": 24,
            "storage_bytes": 60,
            "rdma_bytes": 108,
        },
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_fused_resident_stats(
        "load",
        {
            "resident_hit_entries": 1,
            "admitted_entries": 2,
            "reject_no_victim_entries": 1,
            "evictions": 2,
            "urgent_resident_hit_entries": 1,
            "urgent_admitted_entries": 1,
            "urgent_candidate_evictions": 1,
            "resident_bytes": 36,
            "staging_bytes": 12,
            "storage_bytes": 24,
            "rdma_bytes": 48,
        },
    )
    stats2.record_fused_resident_stats(
        "store",
        {
            "admitted_entries": 5,
            "scratch_bytes": 20,
            "rdma_bytes": 60,
        },
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["fused_resident_load_resident_hit_entries"] == 4
    assert reduced["fused_resident_load_admitted_entries"] == 6
    assert reduced["fused_resident_load_reject_frequency_entries"] == 2
    assert reduced["fused_resident_load_reject_no_victim_entries"] == 1
    assert reduced["fused_resident_load_evictions"] == 2
    assert reduced["fused_resident_load_urgent_resident_hit_entries"] == 3
    assert reduced["fused_resident_load_urgent_admitted_entries"] == 4
    assert reduced["fused_resident_load_urgent_candidate_evictions"] == 2
    assert reduced["fused_resident_load_resident_bytes"] == 120
    assert reduced["fused_resident_load_staging_bytes"] == 36
    assert reduced["fused_resident_load_storage_bytes"] == 84
    assert reduced["fused_resident_load_rdma_bytes"] == 156
    assert reduced["fused_resident_store_admitted_entries"] == 5
    assert reduced["fused_resident_store_scratch_bytes"] == 20


@pytest.mark.parametrize(
    "operation,stats",
    [
        ("promotion", {"admitted_entries": 1}),
        ("load", {"unknown": 1}),
        ("load", {"admitted_entries": -1}),
        ("load", {"admitted_entries": 1.5}),
    ],
)
def test_fused_resident_stats_reject_invalid_values(operation, stats):
    with pytest.raises(ValueError):
        OffloadingConnectorStats().record_fused_resident_stats(
            operation, stats
        )


def test_aggregate_fused_load_stage_latency_stats():
    """Rank-local fused LOAD stage samples aggregate across workers."""
    stats1 = OffloadingConnectorStats()
    stats1.record_fused_load_stage_latency_stats(
        {
            "secondary_to_primary": [0.1],
            "primary_to_gpu": [0.2],
            "dpu_total": [0.3],
            "command_round_trip": [0.31],
            "gpu_visibility": [0.01],
            "worker_total": [0.33],
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_fused_load_stage_latency_stats(
        {
            # A Primary-resident LOAD intentionally omits this stage.
            "primary_to_gpu": [0.4],
            "dpu_total": [0.4],
            "command_round_trip": [0.42],
            "gpu_visibility": [0.02],
            "worker_total": [0.45],
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["fused_load_secondary_to_primary_latency_count"] == 1
    assert reduced["fused_load_secondary_to_primary_latency_sum"] == 0.1
    assert reduced["fused_load_primary_to_gpu_latency_count"] == 2
    assert abs(reduced["fused_load_primary_to_gpu_latency_sum"] - 0.6) < 1e-12
    assert reduced["fused_load_dpu_total_latency_count"] == 2
    assert abs(reduced["fused_load_dpu_total_latency_sum"] - 0.7) < 1e-12
    assert reduced["fused_load_command_round_trip_latency_count"] == 2
    assert (
        abs(reduced["fused_load_command_round_trip_latency_sum"] - 0.73)
        < 1e-12
    )
    assert reduced["fused_load_gpu_visibility_latency_count"] == 2
    assert (
        abs(reduced["fused_load_gpu_visibility_latency_sum"] - 0.03)
        < 1e-12
    )
    assert reduced["fused_load_worker_total_latency_count"] == 2
    assert abs(reduced["fused_load_worker_total_latency_sum"] - 0.78) < 1e-12


def test_aggregate_post_store_primary_ready_reuse_stats():
    """Post-STORE_PRIMARY_READY counters and first reuse latency aggregate."""
    stats1 = OffloadingConnectorStats()
    stats1.record_post_store_primary_ready_reuse_stats(
        {
            "counters": {
                "cohort_keys": 10,
                "reused_keys": 4,
                "load_gpu_ready_key_events": 6,
                "evicted_without_reuse_keys": 2,
                "reused_within_100ms_keys": 1,
                "reused_within_1s_keys": 2,
                "reused_within_10s_keys": 4,
                "reused_within_60s_keys": 4,
            },
            "latencies": {"to_first_load_gpu_ready": [0.05, 0.5]},
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_post_store_primary_ready_reuse_stats(
        {
            "counters": {
                "cohort_keys": 3,
                "reused_keys": 1,
                "load_gpu_ready_key_events": 2,
                "reused_within_100ms_keys": 0,
                "reused_within_1s_keys": 0,
                "reused_within_10s_keys": 1,
                "reused_within_60s_keys": 1,
            },
            "latencies": {"to_first_load_gpu_ready": [2.0]},
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["post_store_primary_ready_cohort_keys"] == 13
    assert reduced["post_store_primary_ready_reused_keys"] == 5
    assert reduced["post_store_primary_ready_load_gpu_ready_key_events"] == 8
    assert reduced["post_store_primary_ready_evicted_without_reuse_keys"] == 2
    assert reduced["post_store_primary_ready_reused_within_100ms_keys"] == 1
    assert reduced["post_store_primary_ready_reused_within_1s_keys"] == 2
    assert reduced["post_store_primary_ready_reused_within_10s_keys"] == 5
    assert reduced["post_store_primary_ready_reused_within_60s_keys"] == 5
    assert (
        reduced[
            "post_store_primary_ready_to_first_load_gpu_ready_latency_count"
        ]
        == 3
    )
    assert (
        reduced[
            "post_store_primary_ready_to_first_load_gpu_ready_latency_sum"
        ]
        == 2.55
    )


def test_aggregate_post_load_primary_published_cross_request_reuse_stats():
    """The origin LOAD is separate from later cross-request reuse."""
    stats1 = OffloadingConnectorStats()
    stats1.record_post_load_primary_published_cross_request_reuse_stats(
        {
            "counters": {
                "cohort_keys": 10,
                "origin_load_gpu_ready_key_events": 9,
                "reused_keys": 3,
                "reuse_load_gpu_ready_key_events": 5,
                "evicted_without_reuse_keys": 6,
                "reused_within_1s_keys": 2,
                "reused_within_10s_keys": 3,
            },
            "latencies": {
                "load_primary_published_to_first_cross_request_"
                "load_gpu_ready": [0.5, 2.0]
            },
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_post_load_primary_published_cross_request_reuse_stats(
        {
            "counters": {
                "cohort_keys": 2,
                "origin_load_gpu_ready_key_events": 2,
                "reused_keys": 1,
                "reuse_load_gpu_ready_key_events": 1,
                "reused_within_1s_keys": 1,
                "reused_within_10s_keys": 1,
            },
            "latencies": {
                "load_primary_published_to_first_cross_request_"
                "load_gpu_ready": [0.25]
            },
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    prefix = "post_load_primary_published_cross_request"
    assert reduced[f"{prefix}_cohort_keys"] == 12
    assert reduced[f"{prefix}_origin_load_gpu_ready_key_events"] == 11
    assert reduced[f"{prefix}_reused_keys"] == 4
    assert reduced[f"{prefix}_reuse_load_gpu_ready_key_events"] == 6
    assert reduced[f"{prefix}_evicted_without_reuse_keys"] == 6
    assert reduced[f"{prefix}_reused_within_1s_keys"] == 3
    assert reduced[f"{prefix}_reused_within_10s_keys"] == 4
    assert (
        reduced[
            "post_load_primary_published_to_first_cross_request_"
            "load_gpu_ready_latency_count"
        ]
        == 3
    )
    assert (
        reduced[
            "post_load_primary_published_to_first_cross_request_"
            "load_gpu_ready_latency_sum"
        ]
        == 2.75
    )


def test_aggregate_secondary_tier_task_runtime_stats():
    """Test that secondary tier task CPU/wall time counters are summed."""
    stats1 = OffloadingConnectorStats()
    stats1.record_secondary_tier_task_runtime_stats(
        {
            "store": {
                "succeeded": {
                    "cpu_seconds": 1.5,
                    "wall_seconds": 4.0,
                    "count": 3,
                },
            },
            "promotion": {
                "succeeded": {
                    "cpu_seconds": 0.5,
                    "wall_seconds": 2.0,
                    "count": 1,
                },
            },
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_secondary_tier_task_runtime_stats(
        {
            "store": {
                "succeeded": {
                    "cpu_seconds": 0.25,
                    "wall_seconds": 1.0,
                    "count": 2,
                },
                "failed": {
                    "cpu_seconds": 0.1,
                    "wall_seconds": 0.4,
                    "count": 1,
                },
            },
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["secondary_tier_store_succeeded_task_cpu_seconds"] == 1.75
    assert reduced["secondary_tier_store_succeeded_task_wall_seconds"] == 5.0
    assert reduced["secondary_tier_store_succeeded_task_count"] == 5
    assert reduced["secondary_tier_store_failed_task_cpu_seconds"] == 0.1
    assert reduced["secondary_tier_store_failed_task_wall_seconds"] == 0.4
    assert reduced["secondary_tier_store_failed_task_count"] == 1
    assert reduced["secondary_tier_promotion_succeeded_task_cpu_seconds"] == 0.5
    assert reduced["secondary_tier_promotion_succeeded_task_wall_seconds"] == 2.0
    assert reduced["secondary_tier_promotion_succeeded_task_count"] == 1


def test_aggregate_tiering_lookup_stats():
    """Test that tiering lookup counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_tiering_lookup_stats(
        {
            "total_blocks": 4,
            "cpu_lookup_blocks": 4,
            "cpu_hit_ready_blocks": 1,
            "cpu_hit_pending_blocks": 1,
            "secondary_lookup_blocks": 2,
            "secondary_hit_ready_blocks": 2,
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_tiering_lookup_stats(
        {
            "total_blocks": 3,
            "secondary_hit_pending_blocks": 1,
            "secondary_hit_but_no_primary_slot_blocks": 1,
            "miss_blocks": 2,
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["tiering_lookup_total_blocks"] == 7
    assert reduced["tiering_lookup_cpu_lookup_blocks"] == 4
    assert reduced["tiering_lookup_cpu_hit_ready_blocks"] == 1
    assert reduced["tiering_lookup_cpu_hit_pending_blocks"] == 1
    assert reduced["tiering_lookup_secondary_lookup_blocks"] == 2
    assert reduced["tiering_lookup_secondary_hit_ready_blocks"] == 2
    assert reduced["tiering_lookup_secondary_hit_pending_blocks"] == 1
    assert reduced["tiering_lookup_secondary_hit_but_no_primary_slot_blocks"] == 1
    assert reduced["tiering_lookup_miss_blocks"] == 2


def test_aggregate_primary_eviction_stats():
    """Test that primary eviction counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_primary_eviction_stats(
        {
            "promotion": {"events": 2, "blocks": 2},
            "store": {"events": 1, "blocks": 3},
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_primary_eviction_stats(
        {
            "promotion": {"events": 1, "blocks": 1},
            "store": {"events": 2, "blocks": 4},
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["primary_eviction_promotion_events"] == 3
    assert reduced["primary_eviction_promotion_blocks"] == 3
    assert reduced["primary_eviction_store_events"] == 3
    assert reduced["primary_eviction_store_blocks"] == 7


def test_reduce():
    """Test that reduce() correctly reduces all nested connector stats."""
    stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
                {"op_size": 16, "op_time": 2},
            ],
        }
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Check that the stats were reduced (should have aggregated values)
    assert "CPU_to_GPU_total_bytes" in reduced
    assert "CPU_to_GPU_total_time" in reduced
    assert "GPU_to_CPU_total_bytes" in reduced
    assert "GPU_to_CPU_total_time" in reduced
    assert reduced["CPU_to_GPU_total_bytes"] == 34
    assert reduced["CPU_to_GPU_total_time"] == 2.6
    assert reduced["GPU_to_CPU_total_time"] == 2.3
    assert reduced["GPU_to_CPU_total_bytes"] == 19


def test_reset():
    """Test that reset() resets all nested connector stats."""
    offload_connector_stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    assert not offload_connector_stats.is_empty()

    offload_connector_stats.reset()

    # After reset, stats should be empty
    assert offload_connector_stats.is_empty()
    assert len(offload_connector_stats.data) == 0
