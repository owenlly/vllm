# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Create file mapper
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        # Scheduler-local presence: keys whose files are known to exist on
        # this tier (a STORE job that completed successfully, or a LOAD, which
        # is only submitted after lookup() saw the file). filter_store_keys()
        # uses it to drop blocks that are already on disk before the tiering
        # orchestrator pins primary blocks and copies them GPU -> primary ->
        # here again; without it the only dedup is the exists() check inside
        # the write task, after the copies have been paid for.
        extra = getattr(offloading_spec, "extra_config", None) or {}
        self._presence_dedup = bool(extra.get("fs_presence_dedup", True))
        self._present: set[OffloadKey] = set()
        self._store_job_keys: dict[int, tuple[OffloadKey, ...]] = {}
        self._presence_skipped_blocks = 0

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> bool | None:
        if key in self._present:
            return True
        return os.path.exists(self.file_mapper.get_file_name(key))

    @override
    def filter_store_keys(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> tuple[OffloadKey, ...]:
        if not self._presence_dedup or not self._present:
            return tuple(keys)
        kept = tuple(k for k in keys if k not in self._present)
        self._presence_skipped_blocks += len(keys) - len(kept)
        return kept

    @property
    def presence_skipped_blocks(self) -> int:
        """Blocks filter_store_keys() dropped because they were already here."""
        return self._presence_skipped_blocks

    @override
    def peek(self, key: OffloadKey) -> bool | None:
        # lookup() on this tier is already a side-effect-free stat().
        return os.path.exists(self.file_mapper.get_file_name(key))

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        primary_kv_view = self._primary_kv_view
        assert primary_kv_view is not None
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._store_job_keys[job_metadata.job_id] = tuple(job_metadata.keys)
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        primary_kv_view = self._primary_kv_view
        assert primary_kv_view is not None
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        # A LOAD is only submitted after lookup() found the file.
        self._present.update(job_metadata.keys)
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue and publish the
        keys of successful STORE jobs as present.
        """
        results = self._pool.get_finished()
        for result in results:
            keys = self._store_job_keys.pop(result.job_id, None)
            if keys is not None and result.success:
                self._present.update(keys)
        return results

    @override
    def quiesce(self) -> None:
        """Wait until all submitted filesystem I/O can be polled as finished."""
        self._pool.wait_idle()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the thread pool, clearing pending tasks and waiting for
        active threads to complete.
        """
        self._pool.shutdown(wait=True)
