"""FastAPI entrypoint for the AI gateway."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import chat

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A minimal gateway that forwards chat messages to Gemini.",
)

app.include_router(chat.router)

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
async def health() -> dict[str, object]:
    """Liveness probe; also reports whether an API key was found."""
    return {
        "status": "ok",
        "model": settings.gemini_model,
        "api_key_configured": settings.is_configured,
    }
