"""Thin wrapper around the Google Gemini API."""

import logging
from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors, types

from app.config import get_settings

logger = logging.getLogger(__name__)

# What the caller is told when the upstream call fails. Deliberately says
# nothing about why: genai_errors.APIError stringifies as
# "<code> <status>. <response_json>", and that response body can quote parts of
# the request back (INVALID_ARGUMENT does), along with model and project
# identifiers. The real reason goes to the log instead.
_UPSTREAM_FAILURE_DETAIL = "Upstream model request failed."


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
        logger.warning("Gemini API error (%s): %s", settings.gemini_model, exc)
        raise LLMError(_UPSTREAM_FAILURE_DETAIL) from exc

    text = (response.text or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text
