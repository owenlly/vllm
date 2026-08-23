# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead JSONL trace for KV placement experiments.

The trace is disabled unless ``VLLM_KV_EVENT_TRACE_PATH`` is set.  Each call
performs one append-only write so scheduler events remain ordered and a
partially completed profiling run is still usable.
"""

from __future__ import annotations

import atexit
from collections.abc import Iterable
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_TRACE_ENV = "VLLM_KV_EVENT_TRACE_PATH"
_lock = threading.Lock()
_trace_fd: int | None = None
_trace_path: str | None = None


def kv_event_trace_enabled() -> bool:
    """Return whether the current process should emit the JSONL trace."""

    return bool(os.environ.get(_TRACE_ENV))


def encode_kv_keys(keys: Iterable[bytes]) -> list[str]:
    """Encode compact byte-string KV keys as stable hexadecimal strings."""

    return [bytes(key).hex() for key in keys]


def external_request_id(request_id: str) -> str:
    """Recover the API-visible ID from vLLM's randomized internal ID."""

    candidate, separator, suffix = request_id.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(ch in "0123456789abcdef" for ch in suffix)
    ):
        return candidate
    return request_id


def emit_kv_event_trace(event: str, /, **fields: Any) -> None:
    """Append one timestamped event when profiling is enabled."""

    path = os.environ.get(_TRACE_ENV)
    if not path:
        return
    record = {
        "schema_version": 1,
        "event": event,
        "wall_time_ns": time.time_ns(),
        "monotonic_time_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    payload = (
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with _lock:
        fd = _get_trace_fd(path)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError("KV event trace write made no progress")
            view = view[written:]


def _get_trace_fd(path: str) -> int:
    global _trace_fd, _trace_path
    if _trace_fd is not None and _trace_path == path:
        return _trace_fd
    _close_trace_fd()
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _trace_fd = os.open(
        trace_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    _trace_path = path
    return _trace_fd


def _close_trace_fd() -> None:
    global _trace_fd, _trace_path
    if _trace_fd is not None:
        os.close(_trace_fd)
    _trace_fd = None
    _trace_path = None


atexit.register(_close_trace_fd)
