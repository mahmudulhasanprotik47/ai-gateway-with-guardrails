"""FastAPI entrypoint for the AI gateway."""

from fastapi import FastAPI

from app.config import get_settings
from app.routers import chat

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A minimal gateway that forwards chat messages to Gemini.",
)

app.include_router(chat.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, object]:
    """Liveness probe; also reports whether an API key was found."""
    return {
        "status": "ok",
        "model": settings.gemini_model,
        "api_key_configured": settings.is_configured,
    }
