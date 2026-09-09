# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class FusedFsLoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for a LOAD whose source blocks are not all in the primary tier yet.

    ``block_ids`` addresses the primary tier exactly like ``CPULoadStoreSpec``,
    so the GPU copy is unchanged. ``fused_indices`` names the positions whose
    bytes still have to be read off the filesystem tier first, and
    ``fused_paths`` gives their files in the same order. The scheduler resolves
    the paths (it owns the FileMapper), so the worker needs no key mapping.

    An empty ``fused_indices`` means every block is already resident and the
    handler degrades to a plain primary→GPU copy.
    """

    @staticmethod
    def medium() -> str:
        return "CPU_FUSED"

    def __init__(
        self,
        block_ids: list[int],
        fused_indices: tuple[int, ...] = (),
        fused_paths: tuple[str, ...] = (),
    ):
        super().__init__(block_ids)
        if len(fused_indices) != len(fused_paths):
            raise ValueError("fused LOAD indices and paths must align")
        num_blocks = len(self.block_ids)
        if any(not 0 <= index < num_blocks for index in fused_indices):
            raise ValueError("fused LOAD index is out of range")
        if len(set(fused_indices)) != len(fused_indices):
            raise ValueError("fused LOAD cannot repeat an index")
        self.fused_indices = tuple(fused_indices)
        self.fused_paths = tuple(fused_paths)

    def __repr__(self) -> str:
        return (
            f"FusedFsLoadStoreSpec(block_ids={self.block_ids!r}, "
            f"fused={len(self.fused_indices)})"
        )
