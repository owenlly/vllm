# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TieringOffloadingManager and ExampleSecondaryTierManager.

These tests verify:
1. Basic tiered offloading operations (store, load, lookup)
2. Cascade behavior (blocks stored to all secondary tiers)
3. Promotion behavior (blocks loaded from secondary to primary to GPU)
4. ref_cnt management (blocks protected during async transfers)
5. Eviction coordination between tiers
"""

from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.common import FusedFsLoadStoreSpec
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="test")
_MOCK_OFFLOADING_SPEC = MagicMock()


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    """Create a mock SharedOffloadRegion for testing."""
    mock = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    mock.create_kv_memoryview.return_value = view
    return mock


def to_keys(int_ids: Iterable[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def count_hits(manager, keys: list[OffloadKey]) -> int | None:
    """Count consecutive lookup hits from the start of keys.

    Returns the count of leading True results, or None if any lookup
    returns None (retry-later signal).
    """
    count = 0
    for key in keys:
        result = manager.lookup(key, _CTX)
        if result is None:
            return None
        if not result:
            break
        count += 1
    return count


def assert_tiering_lookup_stats(
    manager: TieringOffloadingManager,
    expected_counters: dict[str, int],
) -> None:
    stats = manager.take_tiering_lookup_stats()
    for tier in ("primary", "secondary", "total"):
        for clock in ("wall", "cpu"):
            runtime_key = f"{tier}_{clock}_seconds"
            assert runtime_key in stats
            assert stats.pop(runtime_key) >= 0.0
    assert stats == expected_counters


class _DPUBlockIDsLoadStoreSpec(BlockIDsLoadStoreSpec):
    """Block-address spec standing in for a non-CPU primary tier."""

    @staticmethod
    def medium() -> str:
        return "DPU"


class _DPUPrimaryManager(CPUOffloadingManager):
    """CPU policy mechanics with DPU-style worker-facing block metadata."""

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> _DPUBlockIDsLoadStoreSpec:
        keys_list = list(keys)
        blocks_list = list(blocks)
        assert len(keys_list) == len(blocks_list)
        return _DPUBlockIDsLoadStoreSpec(
            [block.block_id for block in blocks_list]
        )


class TestExampleSecondaryTierManager:
    """Tests for ExampleSecondaryTierManager implementation."""

    def test_basic_store_and_lookup(self):
        """Test basic store and lookup operations."""
        mock_view = memoryview(torch.zeros((10, 16), dtype=torch.int8).numpy())
        tier = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
            custom_param=67,
        )

        # Initially empty
        blocks = to_keys(range(3))
        assert tier.lookup(blocks[0], _CTX) is False

        # Store blocks (simulate with direct insertion for testing)
        tier.blocks[blocks[0]] = True
        tier.blocks[blocks[1]] = True

        # Lookup should find first two blocks
        assert tier.lookup(blocks[0], _CTX) is True
        assert tier.lookup(blocks[1], _CTX) is True

        # Third block not present
        assert tier.lookup(blocks[2], _CTX) is False

    def test_factory_accepts_primary_without_memoryview(self):
        tier = SecondaryTierFactory.create_secondary_tier(
            tier_config={"type": "example"},
            primary_kv_view=None,
            offloading_spec=_MOCK_OFFLOADING_SPEC,
        )

        assert isinstance(tier, ExampleSecondaryTierManager)
        assert tier._primary_kv_view is None


def test_dedup_before_primary_skips_the_gpu_copy():
    """Blocks a secondary tier already holds never reach keys_to_store."""
    primary = _DPUPrimaryManager(num_blocks=4)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    held: set = set()
    secondary.filter_store_keys = lambda keys, ctx: tuple(
        k for k in keys if k not in held
    )
    manager = TieringOffloadingManager(
        primary, [secondary], dedup_before_primary=True
    )
    blocks = to_keys(range(3))

    # Nothing known yet: every block is copied.
    out = manager.prepare_store(blocks, _CTX)
    assert out is not None
    assert list(out.keys_to_store) == list(blocks)
    manager.complete_store(blocks, _CTX, success=True)
    manager.reset_cache()

    # The secondary now holds the first two: only the third is copied.
    held.update(blocks[:2])
    out = manager.prepare_store(blocks, _CTX)
    assert out is not None
    assert list(out.keys_to_store) == [blocks[2]]

    # All held: prepare_store returns an empty result, no copy at all.
    held.update(blocks)
    manager.reset_cache()
    out = manager.prepare_store(blocks, _CTX)
    assert out is not None
    assert list(out.keys_to_store) == []


def test_dedup_before_primary_off_by_default():
    """Without the flag the secondary tiers are not consulted up front."""
    primary = _DPUPrimaryManager(num_blocks=4)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    secondary.filter_store_keys = MagicMock(side_effect=lambda keys, ctx: ())
    manager = TieringOffloadingManager(primary, [secondary])
    blocks = to_keys(range(2))

    out = manager.prepare_store(blocks, _CTX)
    assert out is not None
    assert list(out.keys_to_store) == list(blocks)
    secondary.filter_store_keys.assert_not_called()


def test_block_id_primary_supports_cascade_promotion_and_reset():
    """Tiering only requires BlockIDsLoadStoreSpec, not CPU-specific specs."""
    primary = _DPUPrimaryManager(num_blocks=3)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    assert isinstance(secondary, ExampleSecondaryTierManager)
    secondary.quiesce = MagicMock(wraps=secondary.quiesce)
    secondary.reset_cache = MagicMock(wraps=secondary.reset_cache)
    manager = TieringOffloadingManager(primary, [secondary])
    blocks = to_keys(range(2))

    store_output = manager.prepare_store(blocks, _CTX)
    assert store_output is not None
    assert isinstance(store_output.store_spec, _DPUBlockIDsLoadStoreSpec)
    assert store_output.store_spec.block_ids.tolist() == [0, 1]
    manager.complete_store(blocks, _CTX, success=True)

    # reset_cache drains the completed cascade before primary block IDs are
    # released. Persistent secondary data remains available for promotion.
    manager.reset_cache()
    secondary.quiesce.assert_called_once_with()
    secondary.reset_cache.assert_called_once_with()
    assert all(primary.lookup(block, _CTX) is False for block in blocks)
    assert all(secondary.lookup(block, _CTX) is True for block in blocks)

    assert all(manager.lookup(block, _CTX) is None for block in blocks)
    manager.on_schedule_end()  # submit the batched promotion
    manager.on_schedule_end()  # consume its immediate completion

    assert all(primary.lookup(block, _CTX) is True for block in blocks)


class TestTieringOffloadingManager:
    """Tests for TieringOffloadingManager."""

    @pytest.fixture
    def manager_setup(self):
        # Create primary tier (CPU-based)
        mock_region = _mock_mmap_region(5)
        self.primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=mock_region
        )

        mock_view = mock_region.create_kv_memoryview()

        # Create secondary tiers with the primary view
        self.secondary_tier1 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )
        self.secondary_tier2 = ExampleSecondaryTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="example",
        )

        # Create tiered manager. Lookup runtime profiling is off by default in
        # production; these tests assert on the per-tier wall/CPU stats, and
        # one of them turns it back off explicitly.
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary_tier,
            secondary_tiers=[self.secondary_tier1, self.secondary_tier2],
            profile_block_lookup=True,
        )

    def _simulate_on_schedule_end(self):
        """Simulate end of scheduler step: lifecycle flush + drain events."""
        self.manager.on_schedule_end()
        list(self.manager.take_events())

    def test_basic_store_to_primary(self, manager_setup):
        """Test basic store operation to primary tier."""
        blocks = to_keys(range(3))

        # Prepare store
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 3

        # Complete store
        self.manager.complete_store(blocks, _CTX, success=True)

        # Blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

    def test_cascade_to_all_secondary_tiers(self, manager_setup):
        """Test that blocks are cascaded to ALL secondary tiers."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store (triggers cascade via submit_store on each tier)
        self.manager.complete_store(blocks, _CTX, success=True)

        # submit_store was called once per secondary tier
        self.secondary_tier1.submit_store.assert_called_once()
        self.secondary_tier2.submit_store.assert_called_once()

        # Blocks should be in both secondary tiers
        assert self.secondary_tier1.get_num_blocks() == 3
        assert self.secondary_tier2.get_num_blocks() == 3

        # Verify blocks are present
        assert all(self.secondary_tier1.lookup(b, _CTX) for b in blocks)
        assert all(self.secondary_tier2.lookup(b, _CTX) for b in blocks)

    def test_cascade_filters_before_pinning_primary_blocks(self, manager_setup):
        """Known secondary hits are excluded before prepare_load pins slots."""
        blocks = to_keys(range(3))
        keys_to_store = (blocks[1],)
        self.secondary_tier1.filter_store_keys = MagicMock(
            return_value=keys_to_store
        )
        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )

        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        self.secondary_tier1.filter_store_keys.assert_called_once_with(
            blocks, _CTX
        )
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert tuple(job_metadata.keys) == keys_to_store

        # tier1 pins only block 1; tier2 still pins every block.
        assert self.primary_tier._policy.get(blocks[0]).ref_cnt == 1
        assert self.primary_tier._policy.get(blocks[1]).ref_cnt == 2
        assert self.primary_tier._policy.get(blocks[2]).ref_cnt == 1
        assert self.manager.take_secondary_tier_stats() == {
            "store": {
                "candidate_blocks": 6,
                "skipped_present_blocks": 2,
                "submitted_jobs": 2,
                "submitted_blocks": 4,
            }
        }

    def test_cascade_does_not_create_job_when_all_keys_are_filtered(
        self, manager_setup
    ):
        """An all-hit secondary tier creates no empty job or primary pin."""
        blocks = to_keys(range(3))
        self.secondary_tier1.filter_store_keys = MagicMock(return_value=())
        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )

        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        self.secondary_tier1.submit_store.assert_not_called()
        assert all(
            self.primary_tier._policy.get(block).ref_cnt == 1
            for block in blocks
        )
        assert self.manager.take_secondary_tier_stats() == {
            "store": {
                "candidate_blocks": 6,
                "skipped_present_blocks": 3,
                "submitted_jobs": 1,
                "submitted_blocks": 3,
            }
        }

    def test_secondary_tier_stats_track_store_jobs(self, manager_setup):
        """Secondary tier store counters are emitted as deltas."""
        blocks = to_keys(range(3))

        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        assert self.manager.take_secondary_tier_stats() == {
            "store": {
                "candidate_blocks": 6,
                "submitted_jobs": 2,
                "submitted_blocks": 6,
            }
        }
        assert self.manager.take_secondary_tier_stats() == {}

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_stats() == {
            "store": {
                "succeeded_jobs": 2,
                "succeeded_blocks": 6,
            }
        }

    def test_secondary_tier_job_latency_stats_track_store_jobs(
        self, manager_setup
    ):
        """Secondary tier store job latencies are emitted as deltas."""
        blocks = to_keys(range(3))

        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        with patch(
            "vllm.v1.kv_offload.tiering.manager.time.monotonic",
            side_effect=[10.0, 20.0, 11.5, 22.5],
        ):
            self.manager.complete_store(blocks, _CTX, success=True)
            self._simulate_on_schedule_end()
            self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_job_latency_stats() == {
            "store": {
                "succeeded": [1.5, 2.5],
            }
        }
        assert self.manager.take_secondary_tier_job_latency_stats() == {}

    def test_secondary_tier_task_runtime_stats_track_store_jobs(
        self, manager_setup
    ):
        """Secondary tier store task CPU/wall time is emitted as deltas."""
        blocks = to_keys(range(3))

        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)
        for tier, task_cpu_s, task_wall_s in (
            (self.secondary_tier1, 0.6, 2.0),
            (self.secondary_tier2, 0.9, 3.0),
        ):
            completed = tier.completed_jobs[0]
            tier.completed_jobs[0] = JobResult(
                job_id=completed.job_id,
                success=completed.success,
                task_cpu_s=task_cpu_s,
                task_wall_s=task_wall_s,
                task_count=3,
            )

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_task_runtime_stats() == {
            "store": {
                "succeeded": {
                    "cpu_seconds": 1.5,
                    "wall_seconds": 5.0,
                    "count": 6,
                },
            },
        }
        assert self.manager.take_secondary_tier_task_runtime_stats() == {}

    def test_secondary_tier_stats_track_promotion_jobs(self, manager_setup):
        """Secondary tier promotion counters are emitted as deltas."""
        blocks = to_keys(range(3))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        for block in blocks:
            assert self.manager.lookup(block, _CTX) is None

        self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_stats() == {
            "promotion": {
                "submitted_jobs": 1,
                "submitted_blocks": 3,
            }
        }

        self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_stats() == {
            "promotion": {
                "succeeded_jobs": 1,
                "succeeded_blocks": 3,
            }
        }

    def test_secondary_tier_job_latency_stats_track_promotion_jobs(
        self, manager_setup
    ):
        """Secondary tier promotion job latencies are emitted as deltas."""
        blocks = to_keys(range(3))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        for block in blocks:
            assert self.manager.lookup(block, _CTX) is None

        with patch(
            "vllm.v1.kv_offload.tiering.manager.time.monotonic",
            side_effect=[30.0, 31.25, 31.5],
        ):
            self._simulate_on_schedule_end()
            self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_job_latency_stats() == {
            "promotion": {
                "succeeded": [1.25],
            }
        }
        assert self.manager.take_secondary_tier_job_latency_stats() == {}
        timing = self.manager.take_completed_promotion_timing(_CTX.req_id)
        assert timing is not None
        assert timing.submitted_at_s == 30.0
        # Secondary completion was observed at 31.25, but primary publication
        # is sampled only after complete_store() publishes the promoted blocks.
        assert timing.primary_published_at_s == 31.5
        assert self.manager.take_completed_promotion_timing(_CTX.req_id) is None

    def test_secondary_tier_task_runtime_stats_track_promotion_jobs(
        self, manager_setup
    ):
        """Secondary tier promotion task CPU/wall time is emitted as deltas."""
        blocks = to_keys(range(3))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        for block in blocks:
            assert self.manager.lookup(block, _CTX) is None

        self._simulate_on_schedule_end()
        completed = self.secondary_tier1.completed_jobs[0]
        self.secondary_tier1.completed_jobs[0] = JobResult(
            job_id=completed.job_id,
            success=completed.success,
            task_cpu_s=0.25,
            task_wall_s=1.25,
            task_count=3,
        )
        self._simulate_on_schedule_end()

        assert self.manager.take_secondary_tier_task_runtime_stats() == {
            "promotion": {
                "succeeded": {
                    "cpu_seconds": 0.25,
                    "wall_seconds": 1.25,
                    "count": 3,
                },
            },
        }
        assert self.manager.take_secondary_tier_task_runtime_stats() == {}

    def test_tiering_lookup_stats_track_external_outcomes(self, manager_setup):
        """Tiering lookup counters distinguish ready, pending, and miss."""
        cpu_hit, secondary_hit, missing = to_keys(range(3))

        self.manager.prepare_store([cpu_hit], _CTX)
        self.manager.complete_store([cpu_hit], _CTX, success=True)
        self.secondary_tier1.blocks[secondary_hit] = True

        assert self.manager.lookup(cpu_hit, _CTX) is True
        assert self.manager.lookup(secondary_hit, _CTX) is None
        assert self.manager.lookup(missing, _CTX) is False
        assert self.manager.lookup(secondary_hit, _CTX) is None

        assert_tiering_lookup_stats(
            self.manager,
            {
                "total_blocks": 4,
                "cpu_lookup_blocks": 4,
                "cpu_hit_ready_blocks": 1,
                "cpu_hit_pending_blocks": 1,
                "secondary_lookup_blocks": 3,
                "secondary_hit_ready_blocks": 1,
                "miss_blocks": 1,
            },
        )
        assert self.manager.take_tiering_lookup_stats() == {}

    def test_tiering_lookup_runtime_stats_can_be_disabled(self, manager_setup):
        self.manager.profile_block_lookup = False
        missing = to_keys([99])[0]

        assert self.manager.lookup(missing, _CTX) is False

        stats = self.manager.take_tiering_lookup_stats()
        assert stats["total_blocks"] == 1
        assert stats["cpu_lookup_blocks"] == 1
        assert stats["secondary_lookup_blocks"] == 2
        assert stats["miss_blocks"] == 1
        assert not any(name.endswith("_seconds") for name in stats)

    def test_tiering_lookup_stats_track_secondary_pending(self, manager_setup):
        """A pending secondary-tier lookup is counted separately."""
        block = to_keys([99])[0]
        self.secondary_tier1.lookup = MagicMock(return_value=None)
        self.secondary_tier2.lookup = MagicMock(return_value=False)

        assert self.manager.lookup(block, _CTX) is None

        assert_tiering_lookup_stats(
            self.manager,
            {
                "total_blocks": 1,
                "cpu_lookup_blocks": 1,
                "secondary_lookup_blocks": 2,
                "secondary_hit_pending_blocks": 1,
            },
        )

    def test_tiering_lookup_stats_track_pending_before_later_hit(
        self, manager_setup
    ):
        """A pending tier result is counted even if a later tier hits."""
        block = to_keys([99])[0]
        self.secondary_tier1.lookup = MagicMock(return_value=None)
        self.secondary_tier2.blocks[block] = True

        assert self.manager.lookup(block, _CTX) is None

        assert_tiering_lookup_stats(
            self.manager,
            {
                "total_blocks": 1,
                "cpu_lookup_blocks": 1,
                "secondary_lookup_blocks": 2,
                "secondary_hit_pending_blocks": 1,
                "secondary_hit_ready_blocks": 1,
            },
        )

    def test_tiering_lookup_stats_track_no_primary_slot(self, manager_setup):
        """Secondary hit without an evictable primary slot is not a miss."""
        primary_blocks = to_keys(range(5))
        secondary_only_block = to_keys([99])[0]

        result = self.manager.prepare_store(primary_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(primary_blocks, _CTX, success=True)
        self.secondary_tier1.blocks[secondary_only_block] = True

        assert self.manager.lookup(secondary_only_block, _CTX) is False

        assert_tiering_lookup_stats(
            self.manager,
            {
                "total_blocks": 1,
                "cpu_lookup_blocks": 1,
                "secondary_lookup_blocks": 1,
                "secondary_hit_but_no_primary_slot_blocks": 1,
            },
        )

    def test_primary_eviction_stats_track_store_evictions(self, manager_setup):
        """Primary eviction counters include GPU-to-primary store evictions."""
        primary_blocks = to_keys(range(5))
        more_blocks = to_keys(range(5, 7))

        result = self.manager.prepare_store(primary_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(primary_blocks, _CTX, success=True)

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        result = self.manager.prepare_store(more_blocks, _CTX)
        assert result is not None
        assert len(result.evicted_keys) == 2

        assert self.manager.take_primary_eviction_stats() == {
            "store": {
                "events": 1,
                "blocks": 2,
            }
        }
        assert self.manager.take_primary_eviction_stats() == {}

    def test_primary_eviction_stats_track_promotion_evictions(
        self, manager_setup
    ):
        """Primary eviction counters include secondary-to-primary promotions."""
        primary_blocks = to_keys(range(5))
        secondary_only_block = to_keys([99])[0]

        result = self.manager.prepare_store(primary_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(primary_blocks, _CTX, success=True)

        self._simulate_on_schedule_end()
        self._simulate_on_schedule_end()

        self.secondary_tier1.blocks[secondary_only_block] = True

        assert self.manager.lookup(secondary_only_block, _CTX) is None

        assert self.manager.take_primary_eviction_stats() == {
            "promotion": {
                "events": 1,
                "blocks": 1,
            }
        }
        assert self.manager.take_primary_eviction_stats() == {}

    def test_get_usage_stats_forwards_primary_tier(self, manager_setup):
        """Tiering manager reports primary CPU usage to offload metrics."""
        assert self.manager.get_usage_stats() == (0, 5)

        blocks = to_keys(range(2))
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)

        assert self.manager.get_usage_stats() == (2, 5)

    def test_ref_cnt_protection_during_cascade(self, manager_setup):
        """Test that ref_cnt protects blocks during cascade."""
        blocks = to_keys(range(3))

        # Store to primary
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        self.manager.complete_store(blocks, _CTX, success=True)

        # After complete_store, blocks should have ref_cnt > 0
        # (one for each secondary tier)
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            # ref_cnt should be 2 (one for each secondary tier)
            assert block.ref_cnt == 2

        # End of step 1: _maybe_process_finished_jobs() was already called by
        # prepare_store() above (setting the per-step flag), so on_schedule_end()
        # does NOT poll get_finished_jobs() again — cascade completions remain
        # unprocessed until the next step.
        self._simulate_on_schedule_end()

        # ref_cnt still held: cascade jobs finished (sync tier) but haven't
        # been polled yet because the per-step guard skipped the second call.
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 2

        # Secondary tiers have completed jobs waiting to be drained
        assert len(self.secondary_tier1.completed_jobs) > 0
        assert len(self.secondary_tier2.completed_jobs) > 0

        # End of step 2: flag was reset, so _maybe_process_finished_jobs()
        # runs and processes the cascade completions (complete_load → ref_cnt--)
        self._simulate_on_schedule_end()

        # After cascade completes, ref_cnt should be 0
        for block_hash in blocks:
            block = self.primary_tier._policy.get(block_hash)
            assert block.ref_cnt == 0

        # All completed jobs have been drained
        assert len(self.secondary_tier1.completed_jobs) == 0
        assert len(self.secondary_tier2.completed_jobs) == 0

    def test_lookup_from_primary(self, manager_setup):
        """Test lookup when blocks are in primary tier."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)

        # Lookup should find all blocks in primary
        assert count_hits(self.manager, blocks) == 3

    def test_promotion_from_secondary(self, manager_setup):
        """Test promotion of blocks from secondary to primary tier."""
        blocks = to_keys(range(3))

        # Manually add blocks to secondary tier (simulate previous cascade)
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        # Lookup each block to initiate promotion for all of them
        for block in blocks:
            result = self.manager.lookup(block, _CTX)
            assert result is None  # Retry later (promotion initiated)

        # End of step 1: flushes deferred submit_load() calls
        self._simulate_on_schedule_end()

        # End of step 2: processes the completed promotion jobs
        self._simulate_on_schedule_end()

        # Now blocks should be in primary tier
        assert count_hits(self.primary_tier, blocks) == 3

        # Next lookup should succeed
        assert count_hits(self.manager, blocks) == 3

    def test_partial_lookup(self, manager_setup):
        """Test lookup with partial hits."""
        blocks = to_keys(range(5))

        # Store first 3 blocks to primary
        self.manager.prepare_store(blocks[:3], _CTX)
        self.manager.complete_store(blocks[:3], _CTX, success=True)

        # Lookup all 5 blocks should return 3 (first 3 found)
        assert count_hits(self.manager, blocks) == 3

    def test_peek_reports_presence_without_side_effects(self, manager_setup):
        """peek() sees every tier but never promotes or records lookups."""
        primary_only, secondary_only, absent = to_keys(range(3))

        self.manager.prepare_store([primary_only], _CTX)
        self.manager.complete_store([primary_only], _CTX, success=True)
        self.secondary_tier2.blocks[secondary_only] = True
        self.manager.take_tiering_lookup_stats()  # drain the store's lookups

        assert self.manager.peek(primary_only) is True
        assert self.manager.peek(secondary_only) is True
        assert self.manager.peek(absent) is False

        # No promotion was queued for the secondary-only block and the
        # lookup counters did not move.
        assert not self.manager._pending_load_submissions
        assert count_hits(self.primary_tier, [secondary_only]) == 0
        assert self.manager.take_tiering_lookup_stats() == {}

    def test_peek_is_unknown_when_a_tier_cannot_answer(self, manager_setup):
        """A tier without side-effect-free presence yields None on a miss."""
        key = to_keys([0])[0]
        self.secondary_tier1.peek = lambda _key: None  # type: ignore[assignment]

        assert self.manager.peek(key) is None

        self.secondary_tier2.blocks[key] = True
        assert self.manager.peek(key) is True

    def test_eviction_in_primary_tier(self, manager_setup):
        """Test eviction in primary tier when capacity is exceeded."""
        # Primary tier has capacity of 5 blocks
        # First, fill the primary tier
        blocks = to_keys(range(5))
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None
        assert len(result.keys_to_store) == 5
        self.manager.complete_store(blocks, _CTX, success=True)

        # End of step: release ref_cnt from cascade
        self._simulate_on_schedule_end()

        # Now try to store 2 more blocks (should trigger eviction)
        more_blocks = to_keys(range(5, 7))
        result = self.manager.prepare_store(more_blocks, _CTX)

        # Should evict 2 blocks from primary tier
        assert result is not None
        assert len(result.evicted_keys) == 2
        assert len(result.keys_to_store) == 2

    def test_touch_propagates_to_all_tiers(self, manager_setup):
        """Test that touch() propagates to all tiers."""
        blocks = to_keys(range(3))

        # Store blocks
        self.manager.prepare_store(blocks, _CTX)
        self.manager.complete_store(blocks, _CTX, success=True)
        self._simulate_on_schedule_end()

        self.secondary_tier1.touch = MagicMock(wraps=self.secondary_tier1.touch)
        self.secondary_tier2.touch = MagicMock(wraps=self.secondary_tier2.touch)

        # Touch blocks
        self.manager.touch(blocks, _CTX)

        # Verify touch was called on primary tier (check LRU order)
        primary_keys = list(self.primary_tier._policy.blocks.keys())
        assert primary_keys[-3:] == list(reversed(blocks))

        # Verify touch was propagated to all secondary tiers
        self.secondary_tier1.touch.assert_called_once_with(blocks, _CTX)
        self.secondary_tier2.touch.assert_called_once_with(blocks, _CTX)

    def test_failed_store_no_cascade(self, manager_setup):
        """Test that failed GPU→primary store doesn't cascade."""
        blocks = to_keys(range(3))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Prepare store
        result = self.manager.prepare_store(blocks, _CTX)
        assert result is not None

        # Complete store with failure — cascade must not happen
        self.manager.complete_store(blocks, _CTX, success=False)

        # submit_store was never called on either secondary tier
        self.secondary_tier1.submit_store.assert_not_called()
        self.secondary_tier2.submit_store.assert_not_called()

    def test_reset_cache_aborts_deferred_promotions(self, manager_setup):
        block = to_keys([99])[0]
        self.secondary_tier1.blocks[block] = True

        assert self.manager.lookup(block, _CTX) is None
        assert self.primary_tier.lookup(block, _CTX) is None
        assert self.manager._pending_load_submissions

        self.manager.reset_cache()

        assert not self.manager._pending_load_submissions
        assert self.primary_tier.lookup(block, _CTX) is False

    def test_reset_cache_rejects_unquiesced_secondary_job(self, manager_setup):
        block = to_keys([1])[0]
        # This tier accepts the store but never reports its completion.
        self.secondary_tier1.submit_store = MagicMock()

        result = self.manager.prepare_store([block], _CTX)
        assert result is not None
        self.manager.complete_store([block], _CTX, success=True)

        with pytest.raises(RuntimeError, match="secondary jobs remain in flight"):
            self.manager.reset_cache()

        # The slot remains allocated and pinned by the unfinished transfer;
        # reset must not make its block ID available for reuse.
        assert self.primary_tier.lookup(block, _CTX) is True
        assert self.primary_tier._policy.get(block).ref_cnt == 1
        assert len(self.manager._transfer_jobs) == 1

        # The retained job can still finish, after which a retry is safe.
        job_id = next(iter(self.manager._transfer_jobs))
        self.secondary_tier1.completed_jobs.append(
            JobResult(job_id=job_id, success=True)
        )
        self.manager.reset_cache()
        assert not self.manager._transfer_jobs
        assert self.primary_tier.get_usage_stats() == (0, 5)

    def test_lookup_batches_submit_load_per_request(self, manager_setup):
        """lookup() defers submit_load until on_schedule_end(), one per request.

        Blocks from different requests each get their own submit_load call, each
        carrying the correct req_context.
        """
        blocks = to_keys(range(4))
        for block in blocks:
            self.secondary_tier1.blocks[block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        # All lookups return None: secondary hit triggers promotion (in-flight)
        assert self.manager.lookup(blocks[0], ctx_a) is None
        assert self.manager.lookup(blocks[1], ctx_a) is None
        assert self.manager.lookup(blocks[2], ctx_b) is None
        assert self.manager.lookup(blocks[3], ctx_b) is None

        # submit_load must not fire during lookup - only at end of step
        self.secondary_tier1.submit_load.assert_not_called()

        # simulate end of step
        self._simulate_on_schedule_end()

        assert self.secondary_tier1.submit_load.call_count == 2
        calls = self.secondary_tier1.submit_load.call_args_list
        jm_a = calls[0].args[0]
        jm_b = calls[1].args[0]
        assert set(jm_a.keys) == {blocks[0], blocks[1]}
        assert jm_a.req_context is ctx_a
        assert set(jm_b.keys) == {blocks[2], blocks[3]}
        assert jm_b.req_context is ctx_b

    def test_lookup_shared_block_no_duplicate_promotion(self, manager_setup):
        """A block looked up by two requests in the same step is promoted once.

        The first lookup initiates promotion (returns None via secondary hit).
        The second lookup sees ref_cnt=-1 on the primary slot and returns None
        via the primary in-flight path — without triggering a second promotion.
        """
        shared_block = to_keys([0])[0]
        self.secondary_tier1.blocks[shared_block] = True

        self.secondary_tier1.submit_load = MagicMock(
            wraps=self.secondary_tier1.submit_load
        )

        ctx_a = ReqContext(req_id="req_a")
        ctx_b = ReqContext(req_id="req_b")

        result_a = self.manager.lookup(shared_block, ctx_a)
        result_b = self.manager.lookup(shared_block, ctx_b)

        # Both see None (in-flight), but promotion is only queued once
        assert result_a is None
        assert result_b is None

        self._simulate_on_schedule_end()

        # Only one submit_load call despite two lookups
        self.secondary_tier1.submit_load.assert_called_once()
        job_metadata = self.secondary_tier1.submit_load.call_args.args[0]
        assert list(job_metadata.keys) == [shared_block]
        assert job_metadata.req_context is ctx_a

    def test_complete_store_forwards_req_context_to_submit_store(self, manager_setup):
        """complete_store cascades to secondary tiers with the correct req_context."""
        blocks = to_keys(range(2))

        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )

        ctx = ReqContext(req_id="req_ctx", kv_transfer_params={"key": "value"})

        self.manager.prepare_store(blocks, ctx)
        self.manager.complete_store(blocks, ctx, success=True)

        assert self.secondary_tier1.submit_store.call_count == 1
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert job_metadata.req_context is ctx

    def test_on_new_request_lifecycle(self, manager_setup):
        """Policy defaults to BLOCK_LEVEL, escalates when a tier requests it,
        and is cleaned up on on_request_finished."""
        # Default: all tiers return BLOCK_LEVEL
        ctx = ReqContext(req_id="req_policy_lifecycle")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.BLOCK_LEVEL
        self.manager.on_request_finished(ctx)

        # Escalate: tier1 requests REQUEST_LEVEL
        self.secondary_tier1.on_new_request = (
            lambda req_context: RequestOffloadingContext(
                policy=OffloadPolicy.REQUEST_LEVEL
            )
        )

        ctx = ReqContext(req_id="req_policy_lifecycle_2")
        result = self.manager.on_new_request(ctx)
        assert result.policy == OffloadPolicy.REQUEST_LEVEL
        assert ctx.req_id in self.manager._request_level_tiers

        # Cleanup
        self.manager.on_request_finished(ctx)
        assert ctx.req_id not in self.manager._request_level_tiers

    def test_prepare_store_cascades_existing_blocks_to_request_level_tiers(
        self, manager_setup
    ):
        """prepare_store cascades hit blocks to request-level tiers only."""
        # Store some blocks to primary first
        existing_blocks = to_keys(range(3))
        result = self.manager.prepare_store(existing_blocks, _CTX)
        assert result is not None
        self.manager.complete_store(existing_blocks, _CTX, success=True)
        # Drain cascade completions
        self._simulate_on_schedule_end()

        # Make tier1 request-level, tier2 stays block-level
        self.secondary_tier1.on_new_request = (
            lambda req_context: RequestOffloadingContext(
                policy=OffloadPolicy.REQUEST_LEVEL
            )
        )

        ctx = ReqContext(req_id="req_cascade")
        self.manager.on_new_request(ctx)

        # Spy on submit_store
        self.secondary_tier1.submit_store = MagicMock(
            wraps=self.secondary_tier1.submit_store
        )
        self.secondary_tier1.filter_store_keys = MagicMock(
            wraps=self.secondary_tier1.filter_store_keys
        )
        self.secondary_tier2.submit_store = MagicMock(
            wraps=self.secondary_tier2.submit_store
        )

        # Call prepare_store with existing + new blocks
        new_blocks = to_keys(range(3, 5))
        all_blocks = existing_blocks + new_blocks
        result = self.manager.prepare_store(all_blocks, ctx)
        assert result is not None
        assert set(result.keys_to_store) == set(new_blocks)

        # Only tier1 (request-level) should get existing blocks cascaded now.
        # New blocks are cascaded to ALL tiers later via complete_store().
        self.secondary_tier1.submit_store.assert_called_once()
        job_metadata = self.secondary_tier1.submit_store.call_args.args[0]
        assert set(job_metadata.keys) == set(existing_blocks)
        self.secondary_tier1.filter_store_keys.assert_called_once_with(
            tuple(existing_blocks), ctx
        )

        # tier2 (block-level) does not get existing blocks here.
        self.secondary_tier2.submit_store.assert_not_called()


class TestTieringOffloadingWithoutSecondaryTiers:
    """Test TieringOffloadingManager with no secondary tiers (backward compat)."""

    def test_works_without_secondary_tiers(self):
        """Test that manager works with empty secondary_tiers list."""
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=5, mmap_region=_mock_mmap_region(5)
        )

        # Create manager with no secondary tiers
        manager = TieringOffloadingManager(
            primary_tier=primary_tier, secondary_tiers=[]
        )

        blocks = to_keys(range(3))

        # Should work like a regular OffloadingManager
        result = manager.prepare_store(blocks, _CTX)
        assert result is not None
        manager.complete_store(blocks, _CTX, success=True)

        assert count_hits(manager, blocks) == 3

        manager.reset_cache()
        assert count_hits(manager, blocks) == 0


def _fused_manager(num_blocks: int = 4):
    """A tiering manager whose example secondary tier can serve fused LOADs."""
    primary = _DPUPrimaryManager(num_blocks=num_blocks)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    secondary.fused_load_paths = lambda keys: tuple(f"/kv/{k!r}" for k in keys)
    secondary.submit_load = MagicMock()
    manager = TieringOffloadingManager(
        primary, [secondary], fused_secondary_load=True
    )
    return manager, primary, secondary


def test_fused_load_answers_ready_without_a_second_step():
    """A secondary hit is schedulable now, not after a promotion round-trip."""
    manager, primary, secondary = _fused_manager()
    blocks = to_keys(range(2))
    for block in blocks:
        secondary.blocks[block] = True

    # Without fusing this would be None ("promotion started, retry later").
    assert manager.lookup(blocks[0], _CTX) is True
    assert manager.lookup(blocks[1], _CTX) is True
    # No secondary LOAD was submitted: the worker reads the bytes itself.
    secondary.submit_load.assert_not_called()

    # Asking again inside the same request stays ready rather than deferring
    # on the reserved (ref_cnt=-1) primary slot.
    assert manager.lookup(blocks[0], _CTX) is True

    spec = manager.prepare_load(blocks, _CTX)
    assert isinstance(spec, FusedFsLoadStoreSpec)
    assert spec.block_ids.tolist() == [0, 1]
    assert spec.fused_indices == (0, 1)
    assert len(spec.fused_paths) == 2

    # The blocks are only visible once the worker reports the LOAD landed.
    assert primary.lookup(blocks[0], _CTX) is None
    manager.complete_load(set(blocks), _CTX)
    assert primary.lookup(blocks[0], _CTX) is True
    assert primary.lookup(blocks[1], _CTX) is True


def test_fused_load_mixes_reserved_and_resident_blocks():
    """Blocks already in the primary tier keep the ordinary load path."""
    manager, primary, secondary = _fused_manager()
    resident, from_disk = to_keys(range(2))

    out = manager.prepare_store([resident], _CTX)
    assert out is not None
    manager.complete_store([resident], _CTX, success=True)

    secondary.blocks[from_disk] = True
    assert manager.lookup(resident, _CTX) is True
    assert manager.lookup(from_disk, _CTX) is True

    spec = manager.prepare_load([resident, from_disk], _CTX)
    assert isinstance(spec, FusedFsLoadStoreSpec)
    # Only the second block still has to be read off the tier.
    assert spec.fused_indices == (1,)
    assert len(spec.block_ids) == 2


def test_fused_load_failure_releases_the_reserved_slot():
    """A failed LOAD must not publish a block the worker never filled."""
    manager, primary, secondary = _fused_manager(num_blocks=2)
    block = to_keys(range(1))[0]
    secondary.blocks[block] = True

    assert manager.lookup(block, _CTX) is True
    manager.prepare_load([block], _CTX)
    manager.abort_load({block}, _CTX)

    # The slot is free again and the block is not readable.
    assert primary.lookup(block, _CTX) is False


def test_fused_load_releases_slots_when_a_request_ends_early():
    """Reservations made at lookup time do not leak if the request goes away."""
    manager, primary, secondary = _fused_manager(num_blocks=2)
    block = to_keys(range(1))[0]
    secondary.blocks[block] = True

    assert manager.lookup(block, _CTX) is True
    # The request is aborted before prepare_load() is ever reached.
    manager.on_request_finished(_CTX)

    assert primary.lookup(block, _CTX) is False
    # The freed slot can be handed out again.
    assert manager.lookup(block, _CTX) is True


def test_fused_load_off_by_default_keeps_the_two_step_promotion():
    """The default path still defers and submits a secondary LOAD."""
    primary = _DPUPrimaryManager(num_blocks=4)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    manager = TieringOffloadingManager(primary, [secondary])
    block = to_keys(range(1))[0]
    secondary.blocks[block] = True

    assert manager.lookup(block, _CTX) is None


def test_fused_load_requires_tiers_that_can_resolve_paths():
    """A tier that cannot name its files is rejected at construction."""
    primary = _DPUPrimaryManager(num_blocks=2)
    secondary = SecondaryTierFactory.create_secondary_tier(
        tier_config={"type": "example"},
        primary_kv_view=None,
        offloading_spec=_MOCK_OFFLOADING_SPEC,
    )
    with pytest.raises(TypeError, match="fused_load_paths"):
        TieringOffloadingManager(primary, [secondary], fused_secondary_load=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
