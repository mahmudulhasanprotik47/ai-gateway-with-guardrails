"""Thin wrapper around the Google Gemini API."""

from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors, types

from app.config import get_settings


class LLMError(RuntimeError):
    """Raised when the upstream model call cannot be completed."""


@lru_cache
def _get_client() -> genai.Client:
    settings = get_settings()
    if not settings.is_configured:
        raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=settings.google_api_key)


async def generate_reply(message: str) -> str:
    """Send `message` to Gemini and return the model's text reply."""
    settings = get_settings()
    client = _get_client()

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=message,
            # No tools are registered, so skip the SDK's function-calling loop.
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except genai_errors.APIError as exc:
        raise LLMError(f"Gemini API error: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text
