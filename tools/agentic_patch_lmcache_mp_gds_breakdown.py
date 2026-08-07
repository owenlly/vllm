#!/usr/bin/env python3
"""Add stream-accurate LMCache MP GDS LOAD stage metrics."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


EVENT_PATH = Path("v1/mp_observability/event.py")
JOB_LATENCY_PATH = Path(
    "v1/mp_observability/subscribers/metrics/job_latency.py"
)
TRANSFER_PATH = Path("v1/multiprocess/modules/lmcache_driven_transfer.py")


EVENT_REPLACEMENTS = (
    (
        "GDS LOAD event types",
        """    MP_RETRIEVE_START = "mp.retrieve.start"
    MP_RETRIEVE_END = "mp.retrieve.end"
    MP_LOOKUP_PREFETCH_START = "mp.lookup_prefetch.start"
""",
        """    MP_RETRIEVE_START = "mp.retrieve.start"
    MP_RETRIEVE_END = "mp.retrieve.end"
    MP_RETRIEVE_GDS_READ_START = "mp.retrieve.gds_read.start"
    MP_RETRIEVE_GDS_READ_END = "mp.retrieve.gds_read.end"
    MP_RETRIEVE_PAGED_SCATTER_END = "mp.retrieve.paged_scatter.end"
    MP_LOOKUP_PREFETCH_START = "mp.lookup_prefetch.start"
""",
    ),
)


JOB_LATENCY_REPLACEMENTS = (
    (
        "dataclass import",
        """from collections import defaultdict, deque
import threading
from typing import Any
""",
        """from collections import defaultdict, deque
from dataclasses import dataclass
import threading
from typing import Any
""",
    ),
    (
        "GDS LOAD accumulator",
        """class JobLatencyMetricsSubscriber(EventSubscriber):
    \"\"\"Correlate existing MP events without adding work to the data path.\"\"\"
""",
        """@dataclass
class _LoadStageState:
    gds_started: float | None = None
    scatter_started: float | None = None
    gds_read_seconds: float = 0.0
    paged_scatter_seconds: float = 0.0
    saw_stage: bool = False
    valid: bool = True


class JobLatencyMetricsSubscriber(EventSubscriber):
    \"\"\"Correlate existing MP events without adding work to the data path.\"\"\"
""",
    ),
    (
        "GDS LOAD histogram definitions",
        """                "load_end_to_end": (
                    "MP LOAD submitted-to-GPU-stream-end latency."
                ),
                "store_queue": (
""",
        """                "load_end_to_end": (
                    "MP LOAD submitted-to-GPU-stream-end latency."
                ),
                "load_gds_read": (
                    "MP GDS LOAD cumulative slab-to-GPU-staging stream "
                    "latency per request."
                ),
                "load_paged_scatter": (
                    "MP GDS LOAD cumulative GPU-staging-to-paged-KV stream "
                    "latency per request."
                ),
                "load_stage_other": (
                    "MP GDS LOAD stream latency not attributed to GDS reads "
                    "or paged-KV scatter."
                ),
                "store_queue": (
""",
    ),
    (
        "GDS LOAD accumulator storage",
        """        self._active: dict[
            tuple[str, str, str], deque[tuple[float, float | None]]
        ] = defaultdict(deque)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
""",
        """        self._active: dict[
            tuple[str, str, str], deque[tuple[float, float | None]]
        ] = defaultdict(deque)
        self._load_stages: dict[
            tuple[str, str, str], deque[_LoadStageState]
        ] = defaultdict(deque)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
""",
    ),
    (
        "GDS LOAD event subscriptions",
        """            EventType.MP_RETRIEVE_END: (
                lambda event: self._on_end("load", event)
            ),
            EventType.MP_STORE_SUBMITTED: (
""",
        """            EventType.MP_RETRIEVE_END: (
                lambda event: self._on_end("load", event)
            ),
            EventType.MP_RETRIEVE_GDS_READ_START: (
                lambda event: self._on_load_stage("gds_start", event)
            ),
            EventType.MP_RETRIEVE_GDS_READ_END: (
                lambda event: self._on_load_stage("gds_end", event)
            ),
            EventType.MP_RETRIEVE_PAGED_SCATTER_END: (
                lambda event: self._on_load_stage("scatter_end", event)
            ),
            EventType.MP_STORE_SUBMITTED: (
""",
    ),
    (
        "GDS LOAD request start",
        """        self._active[key].append((event.timestamp, submitted))
        self._record(
            f"{operation}_queue", submitted, event.timestamp, event
        )

    def _on_end(self, operation: str, event: Event) -> None:
""",
        """        self._active[key].append((event.timestamp, submitted))
        if operation == "load":
            self._load_stages[key].append(_LoadStageState())
        self._record(
            f"{operation}_queue", submitted, event.timestamp, event
        )

    def _on_load_stage(self, stage: str, event: Event) -> None:
        key = self._key("load", event)
        if key is None:
            return
        states = self._load_stages.get(key)
        if not states:
            return
        state = states[0]
        state.saw_stage = True
        timestamp = event.timestamp
        if stage == "gds_start":
            if state.gds_started is not None or state.scatter_started is not None:
                state.valid = False
                return
            state.gds_started = timestamp
        elif stage == "gds_end":
            if state.gds_started is None or state.scatter_started is not None:
                state.valid = False
                return
            state.gds_read_seconds += max(timestamp - state.gds_started, 0.0)
            state.gds_started = None
            state.scatter_started = timestamp
        elif stage == "scatter_end":
            if state.scatter_started is None or state.gds_started is not None:
                state.valid = False
                return
            state.paged_scatter_seconds += max(
                timestamp - state.scatter_started, 0.0
            )
            state.scatter_started = None
        else:
            state.valid = False

    def _on_end(self, operation: str, event: Event) -> None:
""",
    ),
    (
        "GDS LOAD request end",
        """        self._record(
            f"{operation}_end_to_end",
            submitted,
            event.timestamp,
            event,
        )

    def shutdown(self) -> None:
""",
        """        self._record(
            f"{operation}_end_to_end",
            submitted,
            event.timestamp,
            event,
        )
        if operation == "load":
            states = self._load_stages.get(key)
            state = states.popleft() if states else None
            if states is not None and not states:
                self._load_stages.pop(key, None)
            if state is not None and state.saw_stage:
                state.valid &= (
                    state.gds_started is None
                    and state.scatter_started is None
                )
                if state.valid:
                    total_seconds = max(event.timestamp - started, 0.0)
                    other_seconds = max(
                        total_seconds
                        - state.gds_read_seconds
                        - state.paged_scatter_seconds,
                        0.0,
                    )
                    attrs = self._attrs(event)
                    self._histograms["load_gds_read"].record(
                        state.gds_read_seconds, attributes=attrs
                    )
                    self._histograms["load_paged_scatter"].record(
                        state.paged_scatter_seconds, attributes=attrs
                    )
                    self._histograms["load_stage_other"].record(
                        other_seconds, attributes=attrs
                    )

    def shutdown(self) -> None:
""",
    ),
    (
        "GDS LOAD accumulator cleanup",
        """        self._lookup_starts.clear()
        self._submitted.clear()
        self._active.clear()
""",
        """        self._lookup_starts.clear()
        self._submitted.clear()
        self._active.clear()
        self._load_stages.clear()
""",
    ),
)


TRANSFER_REPLACEMENTS = (
    (
        "standard imports",
        """from dataclasses import dataclass
from itertools import islice
from typing import Generator, Sequence
import threading
import time
""",
        """from dataclasses import dataclass
from itertools import islice
from typing import Any, Generator, Sequence
import os
import threading
import time
""",
    ),
    (
        "GDS LOAD profiling switch",
        """logger = init_logger(__name__)
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = (
""",
        """logger = init_logger(__name__)
_PROFILE_GDS_LOAD_BREAKDOWN = os.getenv(
    "LMCACHE_MP_PROFILE_GDS_LOAD_BREAKDOWN", "false"
).lower() in {"1", "true"}
_HAS_NATIVE_OBJECT_GROUP_TRANSFER: bool = (
""",
    ),
    (
        "transfer helper profiling parameters",
        """def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmc_ops.TransferDirection",
) -> None:
""",
        """def transfer_kv_per_object_group(
    cache_context: BaseCacheContext,
    block_ids_gpu: list[torch.Tensor],
    memory_objs: Sequence[MemoryObj | None],
    object_group_id: int,
    batch_size: int,
    skip_first_n_tokens: int,
    direction: "lmc_ops.TransferDirection",
    load_stage_event_bus: Any | None = None,
    load_stage_stream: Any | None = None,
    load_stage_session_id: str = "",
    load_stage_metadata: dict[str, Any] | None = None,
) -> None:
""",
    ),
    (
        "GDS read stage start",
        """        skip_tokens_in_chunk = effective_start - batch_start_token

        # For H2D, copy from CPU to GPU tmp buffers before the kernel launch
        if is_h2d:
""",
        """        skip_tokens_in_chunk = effective_start - batch_start_token
        profile_gds_load = (
            is_h2d
            and load_stage_event_bus is not None
            and load_stage_stream is not None
            and bool(load_stage_session_id)
            and all(
                isinstance(memory_obj, GDSMemoryObject)
                for memory_obj in memory_object_batch
            )
        )
        if profile_gds_load:
            load_stage_event_bus.publish_on_stream(
                load_stage_stream,
                Event(
                    event_type=EventType.MP_RETRIEVE_GDS_READ_START,
                    session_id=load_stage_session_id,
                    metadata=load_stage_metadata or {},
                ),
            )

        # For H2D, copy from CPU to GPU tmp buffers before the kernel launch
        if is_h2d:
""",
    ),
    (
        "GDS read stage end",
        """                )

        # Do paged KV copy
        for kernel_group_id in kernel_group_ids:
""",
        """                )
        if profile_gds_load:
            load_stage_event_bus.publish_on_stream(
                load_stage_stream,
                Event(
                    event_type=EventType.MP_RETRIEVE_GDS_READ_END,
                    session_id=load_stage_session_id,
                    metadata=load_stage_metadata or {},
                ),
            )

        # Do paged KV copy
        for kernel_group_id in kernel_group_ids:
""",
    ),
    (
        "paged scatter stage end",
        """                recalculated_skip_blocks,
            )

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
""",
        """                recalculated_skip_blocks,
            )
        if profile_gds_load:
            load_stage_event_bus.publish_on_stream(
                load_stage_stream,
                Event(
                    event_type=EventType.MP_RETRIEVE_PAGED_SCATTER_END,
                    session_id=load_stage_session_id,
                    metadata=load_stage_metadata or {},
                ),
            )

        # For D2H, copy from GPU tmp buffers to CPU after the kernel launch
""",
    ),
    (
        "retrieve profiling arguments",
        """                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmc_ops.TransferDirection.H2D,
                        )
""",
        """                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmc_ops.TransferDirection.H2D,
                            load_stage_event_bus=(
                                self._ctx.event_bus
                                if _PROFILE_GDS_LOAD_BREAKDOWN
                                else None
                            ),
                            load_stage_stream=cache_context.cupy_stream,
                            load_stage_session_id=key.request_id,
                            load_stage_metadata={
                                "device": str(cache_context.device),
                                "engine_id": instance_id,
                                "model_name": model_name,
                            },
                        )
""",
    ),
)


LEGACY_REPLACEMENTS = {
    TRANSFER_PATH: (
        (
            "legacy GDS LOAD profiling default",
            '"LMCACHE_MP_PROFILE_GDS_LOAD_BREAKDOWN", "0"\n',
            '"LMCACHE_MP_PROFILE_GDS_LOAD_BREAKDOWN", "false"\n',
        ),
    ),
}


PATCHES = {
    EVENT_PATH: EVENT_REPLACEMENTS,
    JOB_LATENCY_PATH: JOB_LATENCY_REPLACEMENTS,
    TRANSFER_PATH: TRANSFER_REPLACEMENTS,
}

APPLIED_MARKERS = {
    "GDS LOAD profiling switch": "_PROFILE_GDS_LOAD_BREAKDOWN = os.getenv(",
    "GDS read stage end": (
        "event_type=EventType.MP_RETRIEVE_GDS_READ_END"
    ),
    "paged scatter stage end": (
        "event_type=EventType.MP_RETRIEVE_PAGED_SCATTER_END"
    ),
}


def patch_text(
    text: str,
    replacements: tuple[tuple[str, str, str], ...],
    path: Path,
) -> tuple[str, list[str]]:
    applied: list[str] = []
    for description, old, new in replacements:
        marker = APPLIED_MARKERS.get(description)
        if new in text or (marker is not None and marker in text):
            continue
        if old not in text:
            raise RuntimeError(
                f"Could not find expected {description} block in {path}; "
                "refusing to apply a fuzzy patch."
            )
        text = text.replace(old, new, 1)
        applied.append(description)
    return text, applied


def patch_package(root: Path) -> list[str]:
    applied: list[str] = []
    for relative_path, replacements in LEGACY_REPLACEMENTS.items():
        path = root / relative_path
        text = path.read_text()
        original = text
        for description, old, new in replacements:
            if old in text:
                text = text.replace(old, new, 1)
                applied.append(f"{relative_path}: {description}")
        if text != original:
            path.write_text(text)
    for relative_path, replacements in PATCHES.items():
        path = root / relative_path
        text, descriptions = patch_text(
            path.read_text(), replacements, relative_path
        )
        if descriptions:
            path.write_text(text)
            applied.extend(
                f"{relative_path}: {description}"
                for description in descriptions
            )
    return applied


def installed_package_root() -> Path | None:
    spec = importlib.util.find_spec("lmcache")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Patch this LMCache package root instead of the installed package.",
    )
    args = parser.parse_args()
    root = args.root or installed_package_root()
    if root is None:
        print("LMCache not installed; skipping MP GDS LOAD breakdown patch.")
        return

    applied = patch_package(root)
    if not applied:
        print(f"LMCache MP GDS LOAD breakdown patch already applied: {root}")
        return
    print("Applied LMCache MP GDS LOAD breakdown patch:")
    for description in applied:
        print(f"  {description}")


if __name__ == "__main__":
    main()
