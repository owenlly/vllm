"""LMCache MP connector with profiling and exact cache attribution.

The LMCache-provided profiled connector records scheduler-revisit latency but
does not expose the full external prefix hit to vLLM.  That leaves vLLM's
corrected prompt-cache attribution counters at zero.  Keep the profiled
connector as the implementation and add only the missing scheduler-side
attribution hook here.
"""

from typing import TYPE_CHECKING

from lmcache.integration.vllm.profiled_lmcache_mp_connector import (
    LMCacheMPConnector as _ProfiledLMCacheMPConnector,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class LMCacheMPConnector(_ProfiledLMCacheMPConnector):
    """Profiled LMCache MP connector with independent prefix attribution."""

    def get_external_prefix_cache_hit_tokens(
        self, request: "Request"
    ) -> int | None:
        """Return LMCache's full hit, including overlap with vLLM APC."""
        tracker = self._get_request_tracker(request.request_id)
        # LMCache 0.5.2's external MP tracker records the completed lookup in
        # tokens (the builtin vLLM fallback tracker uses blocks instead).
        return tracker.num_lmcache_hit_tokens
