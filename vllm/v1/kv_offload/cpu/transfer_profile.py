# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU-observed KV transfer/forward timestamps, not a GPU timeline."""

import atexit
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def descriptor_adjacency(src, dst, sizes):
    """Summarize consecutive descriptor runs without storing addresses or sorting."""
    import numpy as np

    count = len(sizes)
    if not (len(src) == len(dst) == count):
        raise ValueError("Descriptor array lengths differ")
    if count and np.any(sizes <= 0):
        raise ValueError("Expected positive copy sizes")
    src_next = src[1:] == src[:-1] + sizes[:-1]
    dst_next = dst[1:] == dst[:-1] + sizes[:-1]
    both = src_next & dst_next
    starts = np.flatnonzero(np.r_[True, ~both]) if count else np.array([], int)
    lengths = np.diff(np.r_[starts, count])
    run_bytes = np.add.reduceat(sizes, starts) if count else np.array([], int)
    pairs = np.column_stack((lengths, run_bytes))
    unique, frequencies = np.unique(pairs, axis=0, return_counts=True)
    return {
        "descriptors": count,
        "bytes": int(sizes.sum()),
        "source_adjacent_edges": int(src_next.sum()),
        "destination_adjacent_edges": int(dst_next.sum()),
        "both_adjacent_edges": int(both.sum()),
        "runs": len(starts),
        "bytes_in_multi_descriptor_runs": int(run_bytes[lengths > 1].sum()),
        "run_histogram": [
            {"descriptors": int(pair[0]), "bytes": int(pair[1]), "count": int(n)}
            for pair, n in zip(unique, frequencies)
        ],
    }


class TransferProfiler:
    def __init__(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.file = (path / f"events-{os.getpid()}.jsonl").open("x", buffering=1)
        self.write_ns = 0
        self.records = 0
        self.gpu_markers_enabled = os.getenv("CVM_KV_GPU_MARKERS") == "1"
        self._gpu_pending: list[dict[str, Any]] = []
        self._gpu_sequence = 0
        self._gpu_origin: torch.cuda.Event | None = None
        self.adjacency_enabled = os.getenv("CVM_KV_ADJACENCY_PROFILE") == "1"
        self.emit(
            "metadata",
            schema=1,
            pid=os.getpid(),
            wall_time_ns=time.time_ns(),
            affinity=sorted(os.sched_getaffinity(0)),
            adjacency_enabled=self.adjacency_enabled,
            gpu_markers_enabled=self.gpu_markers_enabled,
            note="CPU monotonic timestamps; completion is observed, not hardware time",
        )
        atexit.register(self.close)

    def gpu_poll(self):
        if not self.gpu_markers_enabled:
            return
        import math

        pending = []
        for ticket in self._gpu_pending:
            if not ticket["end"].query():
                pending.append(ticket)
                continue
            try:
                assert self._gpu_origin is not None
                start_ms = self._gpu_origin.elapsed_time(ticket["start"])
                end_ms = self._gpu_origin.elapsed_time(ticket["end"])
                valid = (
                    math.isfinite(start_ms)
                    and math.isfinite(end_ms)
                    and 0 <= start_ms <= end_ms
                )
                self.emit(
                    "gpu_marker_complete",
                    marker_id=ticket["id"],
                    kind=ticket["kind"],
                    gpu_start_ms=start_ms if math.isfinite(start_ms) else None,
                    gpu_end_ms=end_ms if math.isfinite(end_ms) else None,
                    valid=valid,
                    enqueue_start_ns=ticket["enqueue_ns"],
                    **ticket["tags"],
                )
            except RuntimeError as exc:
                self.emit("gpu_marker_error", marker_id=ticket["id"], error=str(exc))
        self._gpu_pending = pending

    def gpu_begin(self, kind, **tags):
        if not self.gpu_markers_enabled:
            return None
        import torch

        self.gpu_poll()
        if len(self._gpu_pending) >= 1024:
            self.emit("gpu_marker_dropped", kind=kind, reason="pending_limit", **tags)
            return None
        if self._gpu_origin is None:
            self._gpu_origin = torch.cuda.Event(enable_timing=True)
            before = time.monotonic_ns()
            self._gpu_origin.record()
            self._gpu_origin.synchronize()
            self.emit(
                "gpu_clock_origin",
                host_before_ns=before,
                host_after_ns=time.monotonic_ns(),
                note=(
                    "One initial synchronization; subsequent markers do not synchronize"
                ),
            )
        self._gpu_sequence += 1
        ticket = dict(
            id=self._gpu_sequence,
            kind=kind,
            tags=tags,
            start=torch.cuda.Event(enable_timing=True),
            end=torch.cuda.Event(enable_timing=True),
            enqueue_ns=time.monotonic_ns(),
        )
        ticket["start"].record()
        return ticket

    def gpu_end(self, ticket):
        if ticket is None:
            return
        ticket["end"].record()
        self._gpu_pending.append(ticket)
        self.emit(
            "gpu_marker_enqueued",
            marker_id=ticket["id"],
            kind=ticket["kind"],
            enqueue_start_ns=ticket["enqueue_ns"],
            **ticket["tags"],
        )
        self.gpu_poll()

    def adjacency(self, src, dst, sizes):
        if not self.adjacency_enabled:
            return {}
        start = time.monotonic_ns()
        result = descriptor_adjacency(src, dst, sizes)
        return {
            "adjacency": result,
            "adjacency_profile_ns": time.monotonic_ns() - start,
        }

    def emit(self, event: str, **values):
        start = time.monotonic_ns()
        record = dict(
            event=event,
            time_ns=start,
            tid=threading.get_native_id(),
            recorder_write_ns_before=self.write_ns,
            **values,
        )
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.write_ns += time.monotonic_ns() - start
        self.records += 1

    def close(self):
        if not self.file.closed:
            self.gpu_poll()
            if self._gpu_pending:
                self.emit(
                    "gpu_markers_pending_at_close",
                    ids=[t["id"] for t in self._gpu_pending],
                )
            self.emit("recorder_close", write_ns=self.write_ns, records=self.records)
            self.file.close()


@lru_cache(maxsize=1)
def get_transfer_profiler() -> TransferProfiler | None:
    directory = os.getenv("CVM_KV_TRANSFER_PROFILE_DIR")
    return TransferProfiler(directory) if directory else None
