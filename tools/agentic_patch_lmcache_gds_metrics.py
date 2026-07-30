#!/usr/bin/env python3
"""Connect LMCache 0.4.6 GDS I/O to its existing metrics collector."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPLACEMENTS = (
    (
        "statistics import",
        """from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
""",
        """from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
""",
    ),
    (
        "statistics monitor initialization",
        """        self.stats = None  # TODO: plug into LMCache Statistics
""",
        """        self.stats = LMCStatsMonitor.GetOrCreate()
""",
    ),
    (
        "batched read latency",
        """        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results
""",
        """        total_time = time.perf_counter() - start_time
        self.stats.update_interval_remote_time_to_get_sync(total_time * 1000.0)
        retrieve_stats = self.stats.get_current_retrieve_stats()
        if retrieve_stats is not None:
            metric_name = "remote_backend_batched_get_blocking_time"
            retrieve_stats.detailed_metrics[metric_name] = (
                retrieve_stats.detailed_metrics.get(metric_name, 0.0)
                + total_time
            )
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results
""",
    ),
    (
        "cuFile write metrics",
        """                with self.gds_module.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    f.write(
                        addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                    )
""",
        """                with self.gds_module.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    io_start_time = time.perf_counter()
                    write_result = f.write(
                        addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                    )
                    io_time_ms = (
                        time.perf_counter() - io_start_time
                    ) * 1000.0
                    write_bytes = (
                        write_result
                        if isinstance(write_result, int) and write_result >= 0
                        else kv_chunk.nbytes
                    )
                    self.stats.update_interval_remote_write_metrics(write_bytes)
                    self.stats.update_interval_remote_time_to_put(io_time_ms)
""",
    ),
    (
        "cuFile read metrics",
        """                with self.gds_module.CuFile(
                    gds_path, "r", use_direct_io=self.use_direct_io
                ) as f:
                    return f.read(
                        gpu_pointer,
                        size_in_bytes,
                        file_offset=file_offset,
                        dev_offset=dev_offset,
                    )
""",
        """                with self.gds_module.CuFile(
                    gds_path, "r", use_direct_io=self.use_direct_io
                ) as f:
                    io_start_time = time.perf_counter()
                    read_bytes = f.read(
                        gpu_pointer,
                        size_in_bytes,
                        file_offset=file_offset,
                        dev_offset=dev_offset,
                    )
                    io_time_ms = (
                        time.perf_counter() - io_start_time
                    ) * 1000.0
                    if read_bytes > 0:
                        self.stats.update_interval_remote_read_metrics(
                            read_bytes
                        )
                        self.stats.update_interval_remote_time_to_get(
                            io_time_ms
                        )
                    return read_bytes
""",
    ),
)


def patch_text(text: str) -> tuple[str, list[str]]:
    """Apply all known LMCache GDS metric hooks without fuzzy matching."""
    applied = []
    for description, old, new in REPLACEMENTS:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(
                f"Could not find expected {description} block in LMCache "
                "GDS backend; refusing to apply a fuzzy patch."
            )
        text = text.replace(old, new, 1)
        applied.append(description)
    return text, applied


def installed_gds_backend_path() -> Path | None:
    spec = importlib.util.find_spec(
        "lmcache.v1.storage_backend.gds_backend"
    )
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        help="Patch this GDS backend instead of the installed LMCache package.",
    )
    args = parser.parse_args()

    path = args.path or installed_gds_backend_path()
    if path is None:
        print("LMCache GDS backend not installed; skipping metrics patch.")
        return

    text, applied = patch_text(path.read_text())
    if not applied:
        print(f"LMCache GDS metrics patch already applied: {path}")
        return

    path.write_text(text)
    print(
        "Applied LMCache GDS metrics patch "
        f"({', '.join(applied)}): {path}"
    )


if __name__ == "__main__":
    main()
