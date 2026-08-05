# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import (
    IterationStats,
    PrefillStats,
    PromptTokenStats,
    RequestStateStats,
    calculate_lmcache_mp_ttft_breakdown,
)


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    assert repr(iteration_stats).startswith("IterationStats(")


def test_lmcache_mp_ttft_breakdown_closes_external_load_timeline():
    req_stats = RequestStateStats(
        queued_ts=10.0,
        scheduled_ts=25.0,
        external_lookup_started_ts=12.0,
        external_lookup_finished_ts=13.0,
        external_load_gpu_block_wait_started_ts=13.25,
        external_load_admitted_ts=15.0,
        external_load_ready_ts=20.0,
        external_load_revisited_ts=22.0,
    )
    prefill_stats = PrefillStats(
        scheduler_steps=3,
        model_execution_time=10.0,
        scheduler_gap_time=4.0,
    )

    breakdown = calculate_lmcache_mp_ttft_breakdown(
        req_stats,
        first_token_ts=40.0,
        first_token_latency=32.0,
        prefill_stats=prefill_stats,
    )

    assert breakdown["external_load_requests"] == 32.0
    assert breakdown["engine_queue_total"] == 15.0
    assert breakdown["prefill_total"] == 15.0
    assert breakdown["frontend_overhead"] == 2.0
    assert breakdown["queued_before_lookup"] == 2.0
    assert breakdown["external_lookup"] == 1.0
    assert breakdown["lookup_to_load_admission"] == 2.0
    assert breakdown["gpu_block_allocation_wait"] == 1.75
    assert breakdown["external_load"] == 5.0
    assert breakdown["scheduler_revisit"] == 2.0
    assert breakdown["revisit_to_model_schedule"] == 3.0
    assert breakdown["queue_accounting_error"] == 0.0
    assert breakdown["prefill_model_execution"] == 10.0
    assert breakdown["prefill_scheduler_gap"] == 4.0
    assert breakdown["prefill_accounting_error"] == 1.0


def test_lmcache_mp_ttft_breakdown_separates_no_load_requests():
    req_stats = RequestStateStats(
        queued_ts=10.0,
        scheduled_ts=14.0,
        external_lookup_started_ts=11.0,
        external_lookup_finished_ts=12.0,
    )

    breakdown = calculate_lmcache_mp_ttft_breakdown(
        req_stats,
        first_token_ts=20.0,
        first_token_latency=11.0,
        prefill_stats=PrefillStats(
            scheduler_steps=1,
            model_execution_time=6.0,
        ),
    )

    assert breakdown["no_external_load_requests"] == 11.0
    assert "external_load_requests" not in breakdown
    assert "external_load" not in breakdown
    assert breakdown["lookup_to_model_schedule"] == 2.0
    assert breakdown["queue_accounting_error"] == 0.0


def test_lmcache_mp_ttft_breakdown_omits_gpu_wait_for_immediate_allocation():
    req_stats = RequestStateStats(
        queued_ts=10.0,
        scheduled_ts=20.0,
        external_lookup_started_ts=11.0,
        external_lookup_finished_ts=12.0,
        external_load_admitted_ts=13.0,
        external_load_ready_ts=17.0,
        external_load_revisited_ts=18.0,
    )

    breakdown = calculate_lmcache_mp_ttft_breakdown(
        req_stats,
        first_token_ts=24.0,
        first_token_latency=15.0,
        prefill_stats=None,
    )

    assert "gpu_block_allocation_wait" not in breakdown


def test_prefill_kv_computed_with_cache():
    """Test that prefill KV compute correctly excludes cached tokens."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 5.0
    req_stats.num_generation_tokens = 50

    # Case 1: With prefix cache (1200 tokens cached)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-001",
        num_prompt_tokens=10000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=1200,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 10000
    assert finished_req.num_cached_tokens == 1200
    assert finished_req.request_id == "test-req-001"

    # Verify calculation: prefill KV = prompt tokens - cached tokens
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 8800  # 10000 - 1200


def test_prefill_kv_computed_no_cache():
    """Test prefill KV compute without prefix caching."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 10

    # Case 2: No prefix cache
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-002",
        num_prompt_tokens=2000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=0,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 2000
    assert finished_req.num_cached_tokens == 0
    assert finished_req.request_id == "test-req-002"

    # Verify calculation: prefill KV = full prompt when no cache
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 2000


def test_prefill_kv_computed_edge_cases():
    """Test edge cases for prefill KV compute calculation."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 1.0
    req_stats.num_generation_tokens = 1

    # Case 3: Negative num_cached_tokens (shouldn't happen, but handle gracefully)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-003",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=-1,
    )

    finished_req = iteration_stats.finished_requests[0]
    # max() should handle negative values
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 100  # Should treat negative as 0
    assert finished_req.request_id == "test-req-003"

    # Case 4: All tokens cached (shouldn't happen in practice)
    iteration_stats2 = IterationStats()
    iteration_stats2.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-004",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=100,
    )

    finished_req2 = iteration_stats2.finished_requests[0]
    prefill_kv_computed2 = finished_req2.num_prompt_tokens - max(
        finished_req2.num_cached_tokens, 0
    )
    assert prefill_kv_computed2 == 0  # All cached, nothing computed
    assert finished_req2.request_id == "test-req-004"


def test_prompt_token_stats_all_computed():
    """Test all tokens computed locally, no caching."""
    stats = PromptTokenStats()

    # Case 1: No caching (All tokens computed locally)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1000
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 0
    assert stats.total == 1000


def test_prompt_token_stats_partial_local_cache():
    """Test partial local prefix cache hit."""
    stats = PromptTokenStats()

    # Case 2: Partial local cache
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=300,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 700
    assert stats.local_cache_hit == 300
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 300
    assert stats.total == 1000


def test_prompt_token_stats_partial_external_transfer():
    """Test partial external KV transfer."""
    stats = PromptTokenStats()

    # Case 3: Partial external transfer
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=500,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 500
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 500
    assert stats.cached_tokens == 500
    assert stats.total == 1000


def test_prompt_token_stats_mixed_sources():
    """Test mix of local cache and external transfer."""
    stats = PromptTokenStats()

    # Case 4: Mixed sources
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=400,
        num_external_cached_tokens=200,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200
    assert stats.cached_tokens == 600
    assert stats.total == 1000


def test_prompt_cache_attribution_uses_first_local_lookup():
    """Scheduler retries must not reclassify external hits as initial HBM."""
    stats = PromptTokenStats()
    prefill_stats = PrefillStats()

    prefill_stats.observe_initial_local_cache_hit(400)
    # A retry sees a larger HBM prefix, but the first observation is stable.
    prefill_stats.observe_initial_local_cache_hit(700)
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=700,
        num_external_cached_tokens=100,
        num_external_lookup_cached_tokens=800,
    )
    stats.update_from_output(prefill_stats)

    assert stats.attribution_prompt_tokens == 1000
    assert stats.initial_hbm_hit == 400
    assert stats.external_lookup_hit == 800
    assert stats.hbm_external_overlap == 400
    assert stats.external_unique_hit == 400
    assert stats.corrected_total_hit == 800
    assert stats.corrected_compute == 200
    assert stats.external_load == 100


def test_prompt_cache_attribution_requires_independent_external_lookup():
    """Do not guess external overlap from incremental load tokens."""
    stats = PromptTokenStats()
    prefill_stats = PrefillStats()
    prefill_stats.observe_initial_local_cache_hit(400)
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=700,
        num_external_cached_tokens=100,
    )

    stats.update_from_output(prefill_stats)

    assert stats.attribution_prompt_tokens == 0
    assert stats.corrected_total_hit == 0


def test_prompt_token_stats_full_local_cache_recompute():
    """Test full local cache triggers last token recomputation.

    When all tokens are cached, the scheduler forces the model to recompute
    the last token (num_computed_tokens=1), with the rest from cache.
    """
    stats = PromptTokenStats()

    # Case 5: Full local cache (999 cached, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=999,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 999
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 999
    assert stats.total == 1000


def test_prompt_token_stats_full_external_transfer_recompute():
    """Test full external transfer triggers last token recomputation."""
    stats = PromptTokenStats()

    # Case 6: Full external transfer (999 from external, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=999,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 999
    assert stats.cached_tokens == 999
    assert stats.total == 1000
