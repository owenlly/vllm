# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.worker.worker import TransferType

logger = init_logger(__name__)

CPU_CACHE_USAGE_KEY = "cpu_cache_usage"
SECONDARY_TIER_STATS_KEY = "secondary_tier_stats"
SECONDARY_TIER_JOB_LATENCY_STATS_KEY = "secondary_tier_job_latency_stats"
SECONDARY_TIER_TASK_RUNTIME_STATS_KEY = "secondary_tier_task_runtime_stats"
TIERING_LOOKUP_STATS_KEY = "tiering_lookup_stats"
PRIMARY_EVICTION_STATS_KEY = "primary_eviction_stats"
EXTERNAL_LOAD_STAGE_LATENCY_STATS_KEY = "external_load_stage_latency_stats"
RDMA_TRANSPORT_STATS_KEY = "rdma_transport_stats"
RDMA_TRANSPORT_STAT_NAMES = (
    "logical_requests",
    "wr_posted",
    "sge_posted",
    "multi_sge_wr",
    "wr_split",
)


@dataclass
class OffloadingOperationMetrics:
    op_size: int
    op_time: float


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                if k == CPU_CACHE_USAGE_KEY:
                    self.data[k] = v
                    continue
                if k == SECONDARY_TIER_STATS_KEY:
                    self._aggregate_secondary_tier_stats(v)
                    continue
                if k == SECONDARY_TIER_JOB_LATENCY_STATS_KEY:
                    self._aggregate_secondary_tier_job_latency_stats(v)
                    continue
                if k == SECONDARY_TIER_TASK_RUNTIME_STATS_KEY:
                    self._aggregate_secondary_tier_task_runtime_stats(v)
                    continue
                if k == TIERING_LOOKUP_STATS_KEY:
                    self._aggregate_tiering_lookup_stats(v)
                    continue
                if k == PRIMARY_EVICTION_STATS_KEY:
                    self._aggregate_primary_eviction_stats(v)
                    continue
                if k == EXTERNAL_LOAD_STAGE_LATENCY_STATS_KEY:
                    self._aggregate_external_load_stage_latency_stats(v)
                    continue
                if k == RDMA_TRANSPORT_STATS_KEY:
                    self._aggregate_rdma_transport_stats(v)
                    continue
                if k not in self.data:
                    self.data[k] = v
                else:
                    accumulator = self.data[k]
                    assert isinstance(accumulator, list)
                    accumulator.extend(v)
        return self

    def _aggregate_secondary_tier_stats(self, stats: dict[str, Any]) -> None:
        accumulator = self.data.setdefault(SECONDARY_TIER_STATS_KEY, {})
        assert isinstance(accumulator, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            output_operation_stats = accumulator.setdefault(operation, {})
            assert isinstance(output_operation_stats, dict)
            for name, value in operation_stats.items():
                output_operation_stats[name] = (
                    output_operation_stats.get(name, 0) + value
                )

    def _aggregate_secondary_tier_job_latency_stats(
        self, stats: dict[str, Any]
    ) -> None:
        accumulator = self.data.setdefault(
            SECONDARY_TIER_JOB_LATENCY_STATS_KEY, {}
        )
        assert isinstance(accumulator, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            output_operation_stats = accumulator.setdefault(operation, {})
            assert isinstance(output_operation_stats, dict)
            for status, latencies in operation_stats.items():
                output_latencies = output_operation_stats.setdefault(status, [])
                assert isinstance(output_latencies, list)
                output_latencies.extend(latencies)

    def _aggregate_secondary_tier_task_runtime_stats(
        self, stats: dict[str, Any]
    ) -> None:
        accumulator = self.data.setdefault(
            SECONDARY_TIER_TASK_RUNTIME_STATS_KEY, {}
        )
        assert isinstance(accumulator, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            output_operation_stats = accumulator.setdefault(operation, {})
            assert isinstance(output_operation_stats, dict)
            for status, runtime_stats in operation_stats.items():
                assert isinstance(runtime_stats, dict)
                output_runtime_stats = output_operation_stats.setdefault(status, {})
                assert isinstance(output_runtime_stats, dict)
                for name, value in runtime_stats.items():
                    output_runtime_stats[name] = (
                        output_runtime_stats.get(name, 0.0) + value
                    )

    def _aggregate_tiering_lookup_stats(
        self, stats: dict[str, int | float]
    ) -> None:
        accumulator = self.data.setdefault(TIERING_LOOKUP_STATS_KEY, {})
        assert isinstance(accumulator, dict)
        for name, value in stats.items():
            accumulator[name] = accumulator.get(name, 0) + value

    def _aggregate_primary_eviction_stats(self, stats: dict[str, Any]) -> None:
        accumulator = self.data.setdefault(PRIMARY_EVICTION_STATS_KEY, {})
        assert isinstance(accumulator, dict)
        for cause, cause_stats in stats.items():
            assert isinstance(cause_stats, dict)
            output_cause_stats = accumulator.setdefault(cause, {})
            assert isinstance(output_cause_stats, dict)
            for name, value in cause_stats.items():
                output_cause_stats[name] = output_cause_stats.get(name, 0) + value

    def _aggregate_external_load_stage_latency_stats(
        self, stats: dict[str, Any]
    ) -> None:
        accumulator = self.data.setdefault(
            EXTERNAL_LOAD_STAGE_LATENCY_STATS_KEY, {}
        )
        assert isinstance(accumulator, dict)
        for stage, latencies in stats.items():
            output_latencies = accumulator.setdefault(stage, [])
            assert isinstance(output_latencies, list)
            output_latencies.extend(latencies)

    def _aggregate_rdma_transport_stats(
        self, stats: dict[str, Any]
    ) -> None:
        accumulator = self.data.setdefault(RDMA_TRANSPORT_STATS_KEY, {})
        assert isinstance(accumulator, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            output_operation_stats = accumulator.setdefault(operation, {})
            assert isinstance(output_operation_stats, dict)
            for name, value in operation_stats.items():
                output_operation_stats[name] = (
                    output_operation_stats.get(name, 0) + value
                )

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for transfer_type, ops_list in self.data.items():
            if transfer_type == CPU_CACHE_USAGE_KEY:
                assert isinstance(ops_list, dict)
                return_dict["cpu_cache_usage_perc"] = ops_list["usage_perc"]
                return_dict["cpu_cache_used_blocks"] = ops_list["used_blocks"]
                return_dict["cpu_cache_total_blocks"] = ops_list["total_blocks"]
                continue
            if transfer_type == SECONDARY_TIER_STATS_KEY:
                assert isinstance(ops_list, dict)
                for operation, operation_stats in ops_list.items():
                    assert isinstance(operation_stats, dict)
                    for name, value in operation_stats.items():
                        return_dict[f"secondary_tier_{operation}_{name}"] = value
                continue
            if transfer_type == SECONDARY_TIER_JOB_LATENCY_STATS_KEY:
                assert isinstance(ops_list, dict)
                for operation, operation_stats in ops_list.items():
                    assert isinstance(operation_stats, dict)
                    for status, latencies in operation_stats.items():
                        assert isinstance(latencies, list)
                        return_dict[
                            f"secondary_tier_{operation}_{status}_latency_count"
                        ] = len(latencies)
                        return_dict[
                            f"secondary_tier_{operation}_{status}_latency_sum"
                        ] = sum(latencies)
                continue
            if transfer_type == SECONDARY_TIER_TASK_RUNTIME_STATS_KEY:
                assert isinstance(ops_list, dict)
                for operation, operation_stats in ops_list.items():
                    assert isinstance(operation_stats, dict)
                    for status, runtime_stats in operation_stats.items():
                        assert isinstance(runtime_stats, dict)
                        for name, value in runtime_stats.items():
                            return_dict[
                                f"secondary_tier_{operation}_{status}_task_{name}"
                            ] = value
                continue
            if transfer_type == TIERING_LOOKUP_STATS_KEY:
                assert isinstance(ops_list, dict)
                for name, value in ops_list.items():
                    return_dict[f"tiering_lookup_{name}"] = value
                continue
            if transfer_type == PRIMARY_EVICTION_STATS_KEY:
                assert isinstance(ops_list, dict)
                for cause, cause_stats in ops_list.items():
                    assert isinstance(cause_stats, dict)
                    for name, value in cause_stats.items():
                        return_dict[f"primary_eviction_{cause}_{name}"] = value
                continue
            if transfer_type == EXTERNAL_LOAD_STAGE_LATENCY_STATS_KEY:
                assert isinstance(ops_list, dict)
                for stage, latencies in ops_list.items():
                    assert isinstance(latencies, list)
                    return_dict[
                        f"external_load_{stage}_latency_count"
                    ] = len(latencies)
                    return_dict[
                        f"external_load_{stage}_latency_sum"
                    ] = sum(latencies)
                continue
            if transfer_type == RDMA_TRANSPORT_STATS_KEY:
                assert isinstance(ops_list, dict)
                for operation, operation_stats in ops_list.items():
                    assert isinstance(operation_stats, dict)
                    for name, value in operation_stats.items():
                        return_dict[f"rdma_{operation}_{name}"] = value
                continue
            assert isinstance(ops_list, list)
            total_bytes = 0
            total_time = 0.0
            for op in ops_list:
                assert isinstance(op, dict)
                total_bytes += op["op_size"]
                total_time += op["op_time"]
            return_dict[f"{transfer_type}_total_bytes"] = total_bytes
            return_dict[f"{transfer_type}_total_time"] = total_time
        return return_dict

    def is_empty(self) -> bool:
        return not self.data

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        transfer_type_key = src + "_to_" + dst
        op = OffloadingOperationMetrics(num_bytes, time)
        if transfer_type_key in self.data:
            self.data[transfer_type_key].append(op)
        else:
            self.data[transfer_type_key] = [op]

    def record_cpu_cache_usage(
        self,
        *,
        used_blocks: int,
        total_blocks: int,
    ):
        usage_perc = used_blocks / total_blocks if total_blocks else 0.0
        self.data[CPU_CACHE_USAGE_KEY] = {
            "used_blocks": used_blocks,
            "total_blocks": total_blocks,
            "usage_perc": usage_perc,
        }

    def record_secondary_tier_stats(self, stats: dict[str, dict[str, int]]) -> None:
        if not stats:
            return
        self._aggregate_secondary_tier_stats(stats)

    def record_secondary_tier_job_latency_stats(
        self, stats: dict[str, dict[str, list[float]]]
    ) -> None:
        if not stats:
            return
        self._aggregate_secondary_tier_job_latency_stats(stats)

    def record_secondary_tier_task_runtime_stats(
        self, stats: dict[str, dict[str, dict[str, float]]]
    ) -> None:
        if not stats:
            return
        self._aggregate_secondary_tier_task_runtime_stats(stats)

    def record_tiering_lookup_stats(self, stats: dict[str, int | float]) -> None:
        if not stats:
            return
        self._aggregate_tiering_lookup_stats(stats)

    def record_primary_eviction_stats(self, stats: dict[str, dict[str, int]]) -> None:
        if not stats:
            return
        self._aggregate_primary_eviction_stats(stats)

    def record_external_load_stage_latency_stats(
        self, stats: dict[str, list[float]]
    ) -> None:
        if not stats:
            return
        self._aggregate_external_load_stage_latency_stats(stats)

    def record_rdma_transport_stats(
        self,
        operation: str,
        stats: dict[str, int],
    ) -> None:
        if operation not in {"read", "write"}:
            raise ValueError("RDMA operation must be 'read' or 'write'")
        unknown_names = set(stats).difference(RDMA_TRANSPORT_STAT_NAMES)
        if unknown_names:
            raise ValueError(
                f"unknown RDMA transport counters: {sorted(unknown_names)}"
            )
        if any(value < 0 for value in stats.values()):
            raise ValueError("RDMA transport counters must be non-negative")
        self._aggregate_rdma_transport_stats({operation: stats})


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        # (engine_idx, transfer_type) -> (metric with bounded labels)
        self.histogram_transfer_size: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_bytes: dict[tuple[int, str], PromMetricT] = {}
        self.counter_kv_transfer_time: dict[tuple[int, str], PromMetricT] = {}
        self.counter_secondary_tier_jobs: dict[tuple[int, str, str], PromMetricT] = {}
        self.counter_secondary_tier_blocks: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        self.histogram_secondary_tier_job_latency: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        self.histogram_external_load_stage_latency: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_secondary_tier_task_cpu_seconds: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        self.counter_secondary_tier_task_wall_seconds: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        self.counter_secondary_tier_task_count: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        self.counter_tiering_lookup_total_blocks: dict[int, PromMetricT] = {}
        self.counter_tiering_lookup_result_blocks: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_tiering_lookup_wall_seconds: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_tiering_lookup_cpu_seconds: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_primary_eviction_events: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_primary_eviction_blocks: dict[
            tuple[int, str], PromMetricT
        ] = {}
        self.counter_rdma_transport_stats: dict[
            tuple[int, str, str], PromMetricT
        ] = {}
        gauge_cpu_cache_usage = self._gauge_cls(
            name="vllm:kv_offload_cpu_cache_usage_perc",
            documentation=(
                "CPU KV offload cache usage. 1 means 100 percent usage."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_cpu_cache_usage = {
            idx: gauge_cpu_cache_usage.labels(*values)
            for idx, values in per_engine_labelvalues.items()
        }
        buckets = [  # In bytes
            1e6,
            5e6,
            10e6,
            20e6,
            40e6,
            60e6,
            80e6,
            100e6,
            150e6,
            200e6,
        ]
        latency_buckets = [
            0.0005,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
        ]

        self._counter_kv_bytes = self._counter_cls(
            name="vllm:kv_offload_total_bytes",
            documentation="Number of bytes offloaded by KV connector",
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_kv_transfer_time = self._counter_cls(
            name="vllm:kv_offload_total_time",
            documentation="Total time measured by all KV offloading operations",
            labelnames=labelnames + ["transfer_type"],
        )

        self._counter_secondary_tier_jobs = self._counter_cls(
            name="vllm:kv_offload_secondary_tier_jobs",
            documentation="Number of secondary tier KV offload jobs",
            labelnames=labelnames + ["operation", "status"],
        )

        self._counter_secondary_tier_blocks = self._counter_cls(
            name="vllm:kv_offload_secondary_tier_blocks",
            documentation="Number of blocks in secondary tier KV offload jobs",
            labelnames=labelnames + ["operation", "status"],
        )

        self._histogram_secondary_tier_job_latency = self._histogram_cls(
            name="vllm:kv_offload_secondary_tier_job_latency_seconds",
            documentation="Histogram of secondary tier KV offload job latency.",
            buckets=latency_buckets[:],
            labelnames=labelnames + ["operation", "status"],
        )

        self._histogram_external_load_stage_latency = self._histogram_cls(
            name="vllm:kv_offload_external_load_stage_latency_seconds",
            documentation=(
                "Request-visible latency for each external KV LOAD stage."
            ),
            buckets=latency_buckets[:],
            labelnames=labelnames + ["stage"],
        )

        self._counter_secondary_tier_task_cpu_seconds = self._counter_cls(
            name="vllm:kv_offload_secondary_tier_task_cpu_seconds",
            documentation=(
                "Total worker-thread CPU time spent executing secondary tier "
                "KV block I/O tasks."
            ),
            labelnames=labelnames + ["operation", "status"],
        )

        self._counter_secondary_tier_task_wall_seconds = self._counter_cls(
            name="vllm:kv_offload_secondary_tier_task_wall_seconds",
            documentation=(
                "Total wall time spent executing secondary tier KV block I/O "
                "tasks."
            ),
            labelnames=labelnames + ["operation", "status"],
        )

        self._counter_secondary_tier_task_count = self._counter_cls(
            name="vllm:kv_offload_secondary_tier_task_count",
            documentation="Number of secondary tier KV block I/O tasks.",
            labelnames=labelnames + ["operation", "status"],
        )

        counter_tiering_lookup_total_blocks = self._counter_cls(
            name="vllm:kv_offload_tiering_lookup_total_blocks",
            documentation="Number of tiering KV offload block lookup attempts",
            labelnames=labelnames,
        )
        self.counter_tiering_lookup_total_blocks = {
            idx: counter_tiering_lookup_total_blocks.labels(*values)
            for idx, values in per_engine_labelvalues.items()
        }

        self._counter_tiering_lookup_result_blocks = self._counter_cls(
            name="vllm:kv_offload_tiering_lookup_result_blocks",
            documentation="Number of tiering KV offload block lookup results",
            labelnames=labelnames + ["result"],
        )

        self._counter_tiering_lookup_wall_seconds = self._counter_cls(
            name="vllm:kv_offload_tiering_lookup_wall_seconds",
            documentation=(
                "Total wall time spent in tiering KV offload lookup by tier."
            ),
            labelnames=labelnames + ["tier"],
        )

        self._counter_tiering_lookup_cpu_seconds = self._counter_cls(
            name="vllm:kv_offload_tiering_lookup_cpu_seconds",
            documentation=(
                "Total thread CPU time spent in tiering KV offload lookup by tier."
            ),
            labelnames=labelnames + ["tier"],
        )

        self._counter_primary_eviction_events = self._counter_cls(
            name="vllm:kv_offload_primary_eviction_events",
            documentation="Number of CPU primary tier eviction events by cause",
            labelnames=labelnames + ["cause"],
        )

        self._counter_primary_eviction_blocks = self._counter_cls(
            name="vllm:kv_offload_primary_eviction_blocks",
            documentation="Number of CPU primary tier blocks evicted by cause",
            labelnames=labelnames + ["cause"],
        )

        rdma_counter_docs = {
            "logical_requests": (
                "Number of rank-local logical RDMA transfer ranges."
            ),
            "wr_posted": "Number of RDMA work requests posted to send queues.",
            "sge_posted": (
                "Number of scatter-gather entries carried by posted RDMA "
                "work requests."
            ),
            "multi_sge_wr": (
                "Number of posted RDMA work requests containing multiple SGEs."
            ),
            "wr_split": (
                "Number of extra RDMA work requests caused by splitting a "
                "logical transfer range."
            ),
        }
        self._counter_rdma_transport_stats = {
            name: self._counter_cls(
                name=f"vllm:kv_offload_rdma_{name}",
                documentation=documentation,
                labelnames=labelnames + ["operation"],
            )
            for name, documentation in rdma_counter_docs.items()
        }

        self._histogram_transfer_size = self._histogram_cls(
            name="vllm:kv_offload_size",
            documentation="Histogram of KV offload transfer size, in bytes.",
            buckets=buckets[:],
            labelnames=labelnames + ["transfer_type"],
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Observe transfer statistics from the new data structure.
        transfer_stats_data is expected to be a dict where:
        - keys are transfer type strings (e.g., "cpu_to_gpu", "gpu_to_cpu")
        - values are lists of OffloadingOperationMetrics objects
        """

        for transfer_type, ops in transfer_stats_data.items():
            if transfer_type == CPU_CACHE_USAGE_KEY:
                assert isinstance(ops, dict)
                self.gauge_cpu_cache_usage[engine_idx].set(ops["usage_perc"])
                continue
            if transfer_type == SECONDARY_TIER_STATS_KEY:
                self._observe_secondary_tier_stats(ops, engine_idx)
                continue
            if transfer_type == SECONDARY_TIER_JOB_LATENCY_STATS_KEY:
                self._observe_secondary_tier_job_latency_stats(ops, engine_idx)
                continue
            if transfer_type == SECONDARY_TIER_TASK_RUNTIME_STATS_KEY:
                self._observe_secondary_tier_task_runtime_stats(ops, engine_idx)
                continue
            if transfer_type == TIERING_LOOKUP_STATS_KEY:
                self._observe_tiering_lookup_stats(ops, engine_idx)
                continue
            if transfer_type == PRIMARY_EVICTION_STATS_KEY:
                self._observe_primary_eviction_stats(ops, engine_idx)
                continue
            if transfer_type == EXTERNAL_LOAD_STAGE_LATENCY_STATS_KEY:
                self._observe_external_load_stage_latency_stats(ops, engine_idx)
                continue
            if transfer_type == RDMA_TRANSPORT_STATS_KEY:
                self._observe_rdma_transport_stats(ops, engine_idx)
                continue
            # Cache:
            if (engine_idx, transfer_type) not in self.histogram_transfer_size:
                self.histogram_transfer_size[(engine_idx, transfer_type)] = (
                    self._histogram_transfer_size.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_bytes[(engine_idx, transfer_type)] = (
                    self._counter_kv_bytes.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )
                self.counter_kv_transfer_time[(engine_idx, transfer_type)] = (
                    self._counter_kv_transfer_time.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [transfer_type])
                    )
                )

            # Process ops:
            assert isinstance(ops, list)
            for op in ops:  # ops is a list of serialized OffloadingOperationMetrics
                assert isinstance(op, dict)
                # Observe size histogram
                self.histogram_transfer_size[(engine_idx, transfer_type)].observe(
                    op["op_size"]
                )

                # Increment byte and time counters
                self.counter_kv_bytes[(engine_idx, transfer_type)].inc(op["op_size"])

                self.counter_kv_transfer_time[(engine_idx, transfer_type)].inc(
                    op["op_time"]
                )

    def _observe_rdma_transport_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            for name, value in operation_stats.items():
                counter_key = (engine_idx, operation, name)
                if counter_key not in self.counter_rdma_transport_stats:
                    self.counter_rdma_transport_stats[counter_key] = (
                        self._counter_rdma_transport_stats[name].labels(
                            *(
                                self.per_engine_labelvalues[engine_idx]
                                + [operation]
                            )
                        )
                    )
                if value:
                    self.counter_rdma_transport_stats[counter_key].inc(value)

    def _observe_secondary_tier_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            for name, value in operation_stats.items():
                if name.endswith("_jobs"):
                    status = name.removesuffix("_jobs")
                    counter_key = (engine_idx, operation, status)
                    if counter_key not in self.counter_secondary_tier_jobs:
                        self.counter_secondary_tier_jobs[counter_key] = (
                            self._counter_secondary_tier_jobs.labels(
                                *(
                                    self.per_engine_labelvalues[engine_idx]
                                    + [operation, status]
                                )
                            )
                        )
                    self.counter_secondary_tier_jobs[counter_key].inc(value)
                elif name.endswith("_blocks"):
                    status = name.removesuffix("_blocks")
                    counter_key = (engine_idx, operation, status)
                    if counter_key not in self.counter_secondary_tier_blocks:
                        self.counter_secondary_tier_blocks[counter_key] = (
                            self._counter_secondary_tier_blocks.labels(
                                *(
                                    self.per_engine_labelvalues[engine_idx]
                                    + [operation, status]
                                )
                            )
                        )
                    self.counter_secondary_tier_blocks[counter_key].inc(value)

    def _observe_secondary_tier_job_latency_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            for status, latencies in operation_stats.items():
                counter_key = (engine_idx, operation, status)
                if counter_key not in self.histogram_secondary_tier_job_latency:
                    self.histogram_secondary_tier_job_latency[counter_key] = (
                        self._histogram_secondary_tier_job_latency.labels(
                            *(
                                self.per_engine_labelvalues[engine_idx]
                                + [operation, status]
                            )
                        )
                    )
                assert isinstance(latencies, list)
                for latency_s in latencies:
                    self.histogram_secondary_tier_job_latency[counter_key].observe(
                        latency_s
                    )

    def _observe_external_load_stage_latency_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for stage, latencies in stats.items():
            counter_key = (engine_idx, stage)
            if counter_key not in self.histogram_external_load_stage_latency:
                self.histogram_external_load_stage_latency[counter_key] = (
                    self._histogram_external_load_stage_latency.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [stage])
                    )
                )
            assert isinstance(latencies, list)
            for latency_s in latencies:
                self.histogram_external_load_stage_latency[counter_key].observe(
                    latency_s
                )

    def _observe_secondary_tier_task_runtime_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for operation, operation_stats in stats.items():
            assert isinstance(operation_stats, dict)
            for status, runtime_stats in operation_stats.items():
                assert isinstance(runtime_stats, dict)
                counter_key = (engine_idx, operation, status)
                if counter_key not in self.counter_secondary_tier_task_cpu_seconds:
                    labelvalues = (
                        self.per_engine_labelvalues[engine_idx]
                        + [operation, status]
                    )
                    self.counter_secondary_tier_task_cpu_seconds[counter_key] = (
                        self._counter_secondary_tier_task_cpu_seconds.labels(
                            *labelvalues
                        )
                    )
                    self.counter_secondary_tier_task_wall_seconds[counter_key] = (
                        self._counter_secondary_tier_task_wall_seconds.labels(
                            *labelvalues
                        )
                    )
                    self.counter_secondary_tier_task_count[counter_key] = (
                        self._counter_secondary_tier_task_count.labels(*labelvalues)
                    )

                cpu_seconds = runtime_stats.get("cpu_seconds", 0.0)
                wall_seconds = runtime_stats.get("wall_seconds", 0.0)
                task_count = runtime_stats.get("count", 0.0)
                if cpu_seconds:
                    self.counter_secondary_tier_task_cpu_seconds[counter_key].inc(
                        cpu_seconds
                    )
                if wall_seconds:
                    self.counter_secondary_tier_task_wall_seconds[counter_key].inc(
                        wall_seconds
                    )
                if task_count:
                    self.counter_secondary_tier_task_count[counter_key].inc(task_count)

    def _observe_tiering_lookup_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        total_blocks = stats.get("total_blocks")
        if total_blocks:
            self.counter_tiering_lookup_total_blocks[engine_idx].inc(total_blocks)

        for name, value in stats.items():
            if name == "total_blocks":
                continue
            if name.endswith("_blocks"):
                result = name.removesuffix("_blocks")
                counter_key = (engine_idx, result)
                if counter_key not in self.counter_tiering_lookup_result_blocks:
                    self.counter_tiering_lookup_result_blocks[counter_key] = (
                        self._counter_tiering_lookup_result_blocks.labels(
                            *(self.per_engine_labelvalues[engine_idx] + [result])
                        )
                    )
                self.counter_tiering_lookup_result_blocks[counter_key].inc(value)
            elif name.endswith("_wall_seconds"):
                tier = name.removesuffix("_wall_seconds")
                counter_key = (engine_idx, tier)
                if counter_key not in self.counter_tiering_lookup_wall_seconds:
                    self.counter_tiering_lookup_wall_seconds[counter_key] = (
                        self._counter_tiering_lookup_wall_seconds.labels(
                            *(self.per_engine_labelvalues[engine_idx] + [tier])
                        )
                    )
                self.counter_tiering_lookup_wall_seconds[counter_key].inc(value)
            elif name.endswith("_cpu_seconds"):
                tier = name.removesuffix("_cpu_seconds")
                counter_key = (engine_idx, tier)
                if counter_key not in self.counter_tiering_lookup_cpu_seconds:
                    self.counter_tiering_lookup_cpu_seconds[counter_key] = (
                        self._counter_tiering_lookup_cpu_seconds.labels(
                            *(self.per_engine_labelvalues[engine_idx] + [tier])
                        )
                    )
                self.counter_tiering_lookup_cpu_seconds[counter_key].inc(value)

    def _observe_primary_eviction_stats(
        self, stats: dict[str, Any], engine_idx: int
    ) -> None:
        assert isinstance(stats, dict)
        for cause, cause_stats in stats.items():
            assert isinstance(cause_stats, dict)
            counter_key = (engine_idx, cause)
            if counter_key not in self.counter_primary_eviction_events:
                self.counter_primary_eviction_events[counter_key] = (
                    self._counter_primary_eviction_events.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [cause])
                    )
                )
                self.counter_primary_eviction_blocks[counter_key] = (
                    self._counter_primary_eviction_blocks.labels(
                        *(self.per_engine_labelvalues[engine_idx] + [cause])
                    )
                )

            events = cause_stats.get("events", 0)
            blocks = cause_stats.get("blocks", 0)
            if events:
                self.counter_primary_eviction_events[counter_key].inc(events)
            if blocks:
                self.counter_primary_eviction_blocks[counter_key].inc(blocks)
