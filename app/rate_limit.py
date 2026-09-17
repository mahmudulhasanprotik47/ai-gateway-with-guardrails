"""Per-client request rate limiting for /chat.

Sliding window log: one deque of monotonic timestamps per client. A fixed
window would be smaller code, but it admits twice the limit across a boundary
(``limit`` requests at 59.9s, ``limit`` more at 60.1s), and that 2x burst is
exactly what overruns the upstream quota and produces the confusing Gemini
error this limiter exists to pre-empt.

Scope: state is a module-level dict, so this is correct for exactly one
process. Under ``uvicorn --workers N``, or several instances behind a proxy,
each process keeps its own counters and the effective limit becomes N x the
configured one. Correct multi-process limiting needs a shared store - Redis
with an atomic INCR+EXPIRE, or a sorted set for a real sliding window. Not
built here.

Known gap, documented rather than fixed: Starlette reads and JSON-parses the
request body before dependencies run, so a rejected caller still costs a full
read and parse. This limits work sent to Gemini, not work done by the process;
a Content-Length guard is the follow-up. tests/test_auth.py documents the
sibling case for authentication.
"""

import logging
import math
import time
from collections import deque

from fastapi import Depends, HTTPException, Request, status

from app.auth import AuthenticatedClient, require_api_key
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ponytail: process-local state, single instance only. Swap for Redis
# (INCR+EXPIRE, or a sorted set) if this ever runs as more than one process.
# Keys are key_ids, so entries are bounded by the number of configured
# GATEWAY_API_KEYS: only authenticated requests get this far, and an
# unauthenticated flood cannot grow this dict at all. No eviction needed.
_buckets: dict[str, deque[float]] = {}

# Indirection so tests advance time instead of sleeping. Monotonic, so a clock
# step cannot unlock a bucket early or hold one shut.
_now = time.monotonic


def reset() -> None:
    """Forget every bucket. Used by tests; nothing calls this at runtime."""
    _buckets.clear()


async def enforce_rate_limit(
    request: Request,
    client: AuthenticatedClient = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> None:
    """Admit at most ``rate_limit_requests`` per window, per authenticated key.

    Auth is a sub-dependency rather than a sibling on purpose: FastAPI resolves
    sub-dependencies before the dependent, so a 401 structurally cannot be
    pre-empted by a 429, and the limiter can never run without an identity.
    FastAPI caches dependency results per request, so ``require_api_key`` still
    executes once even though the router lists it as well.

    This MUST stay ``async def``. A sync dependency is run in a threadpool, and
    the length check and the append below would stop being atomic.
    """
    limit = settings.rate_limit_requests
    window = settings.rate_limit_window_seconds
    now = _now()

    bucket = _buckets.setdefault(client.key_id, deque())
    cutoff = now - window
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()

    if len(bucket) >= limit:
        # The rejected request is deliberately NOT appended. Recording it would
        # push the window forward on every retry, so a client under sustained
        # load would never drain its bucket and Retry-After would be a lie.
        # The bucket is empty here only when the limit is 0 (deliberately shut).
        oldest = bucket[0] if bucket else now
        retry_after = max(1, math.ceil(oldest + window - now))
        # key_id is a non-secret label; the key itself never reaches a log.
        logger.warning(
            "Rate limited %s for client %s: over %d requests per %ds",
            request.url.path,
            client.key_id,
            limit,
            window,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            # No key_id in the body: logs are ours, responses are the client's.
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)
