# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
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
    """Actual verbs counters are summed by operation across TP workers."""
    stats1 = OffloadingConnectorStats()
    stats1.record_rdma_transport_stats(
        "write",
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
        "write",
        {
            "logical_requests": 6,
            "wr_posted": 3,
            "sge_posted": 12,
            "multi_sge_wr": 3,
            "wr_split": 1,
        },
    )
    stats2.record_rdma_transport_stats(
        "read",
        {
            "logical_requests": 2,
            "wr_posted": 2,
            "sge_posted": 8,
            "multi_sge_wr": 2,
            "wr_split": 0,
        },
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["rdma_write_logical_requests"] == 16
    assert reduced["rdma_write_wr_posted"] == 7
    assert reduced["rdma_write_sge_posted"] == 32
    assert reduced["rdma_write_multi_sge_wr"] == 7
    assert reduced["rdma_write_wr_split"] == 1
    assert reduced["rdma_read_logical_requests"] == 2
    assert reduced["rdma_read_wr_posted"] == 2
    assert reduced["rdma_read_sge_posted"] == 8
    assert reduced["rdma_read_multi_sge_wr"] == 2
    assert reduced["rdma_read_wr_split"] == 0


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
            "secondary_to_primary": [0.3],
            "scheduler_gap": [0.02],
            "primary_to_gpu": [0.04],
            "end_to_end": [0.36],
        }
    )
    stats2 = OffloadingConnectorStats()
    stats2.record_external_load_stage_latency_stats(
        {
            "secondary_to_primary": [0.5],
            "scheduler_gap": [0.03],
            "primary_to_gpu": [0.07],
            "end_to_end": [0.6],
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["external_load_secondary_to_primary_latency_count"] == 2
    assert reduced["external_load_secondary_to_primary_latency_sum"] == 0.8
    assert reduced["external_load_scheduler_gap_latency_count"] == 2
    assert reduced["external_load_scheduler_gap_latency_sum"] == 0.05
    assert reduced["external_load_primary_to_gpu_latency_count"] == 2
    assert (
        abs(reduced["external_load_primary_to_gpu_latency_sum"] - 0.11)
        < 1e-12
    )
    assert reduced["external_load_end_to_end_latency_count"] == 2
    assert abs(reduced["external_load_end_to_end_latency_sum"] - 0.96) < 1e-12


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
