"""FastAPI entrypoint for the AI gateway."""

import json
import logging
import sys
import time
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.routers import chat

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A minimal gateway that forwards chat messages to Gemini.",
)

app.include_router(chat.router)


class _StdoutHandler(logging.StreamHandler):
    """A StreamHandler that looks up sys.stdout on every write.

    StreamHandler(sys.stdout) would keep whichever stream was current at import
    time; under pytest that is the global capture, so capsys would never see a
    line and a "goes to stdout" test would prove nothing.
    """

    stream = property(lambda self: sys.stdout, lambda self, value: None)


# One JSON line per /chat request, to stdout only: no aggregator, which is out of
# scope for a localhost gateway. uvicorn configures only its own loggers, so
# without a handler here an INFO line would be dropped. propagate stays True so
# caplog sees it; a root handler (--log-config, basicConfig) would print it twice.
_request_log = logging.getLogger("app.request_log")
_request_log.setLevel(logging.INFO)
if not _request_log.handlers:  # a re-import must not double every line
    _request_log.addHandler(_StdoutHandler())

# Indirection so tests control latency instead of sleeping, as in rate_limit.
_now = time.perf_counter

_STANDARD_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"}
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Emit one summary line per /chat request, however it ended.

    Middleware is the only layer that sees every outcome: HTTPExceptions from
    the dependencies and the handler (401, 429, 400, 502) and the 422 handler
    all become responses inside it. It cannot take Depends, so the values it
    reports arrive through ``request.state``, which is a view onto the scope
    dict shared with the handler's Request: ``require_api_key`` sets ``key_id``
    on success and ``chat()`` sets ``model`` once routing returns.

    The line is a closed allowlist of seven fields, none of which can carry
    request or reply content. ``path`` is a constant and ``method`` is clamped,
    and nothing reads headers, the query string, either body or exception text.
    tests/test_request_log.py pins both the keys and the values.
    """
    # scope["path"] is already percent-decoded and has no query string.
    if request.scope["path"] != "/chat":
        return await call_next(request)

    start = _now()
    # Stays 500 if call_next raises; ServerErrorMiddleware, outside us, builds
    # that response. A client disconnect also lands here as 500 - a known gap.
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        method = request.method
        _request_log.info(
            json.dumps(
                {
                    "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "key_id": getattr(request.state, "key_id", "unauthenticated"),
                    "method": method if method in _STANDARD_METHODS else "OTHER",
                    "path": "/chat",
                    "status": status_code,
                    # Until the response starts, not until the body is sent.
                    "latency_ms": round((_now() - start) * 1000, 1),
                    "model": getattr(request.state, "model", None),
                }
            )
        )

# Pydantic records the value it rejected, and FastAPI's default handler
# serializes the error list as-is. For /chat that value is the whole message, so
# an over-long prompt carrying a card number or an email came straight back in
# the 422 body (and made a ~16 KB response out of a rejected request).
_LEAKY_ERROR_FIELDS = frozenset({"input", "ctx"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Report what was wrong with the request without quoting it back.

    `type`, `loc` and `msg` tell the caller which field failed and why, which is
    everything they need to fix it. `input` and `ctx` are dropped because they
    carry the rejected payload itself.
    """
    safe_errors = [
        {key: value for key, value in error.items() if key not in _LEAKY_ERROR_FIELDS}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": safe_errors},
    )


@app.get("/health", tags=["meta"])
async def health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    """Liveness probe; also reports whether an API key was found.

    Settings arrive by dependency rather than from the module-level instance
    above: `get_settings` is `lru_cache`d, so a direct call would ignore
    `app.dependency_overrides` and report whatever was cached first. The
    module-level `settings` is still fine for the app metadata, which is fixed
    at construction.
    """
    # Both tiers: `model` keeps its meaning (the escalation tier), but short
    # messages are answered by `cheap_model`, so reporting only one would
    # mislead about what actually serves a typical request.
    return {
        "status": "ok",
        "model": settings.gemini_model,
        "cheap_model": settings.gemini_cheap_model,
        "api_key_configured": settings.is_configured,
    }
