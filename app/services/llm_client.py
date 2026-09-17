"""Thin wrapper around the Google Gemini API."""

import logging
from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors, types

from app.config import get_settings

logger = logging.getLogger(__name__)

# Every message raised below is for operators, not callers: the router replaces
# it with one fixed string before answering. genai_errors.APIError stringifies
# as "<code> <status>. <response_json>", and that body can quote the request
# back (INVALID_ARGUMENT does) along with model and project identifiers, so the
# upstream text is logged and never carried in an exception message either.

# Explicit rather than inherited: without this the posture is whatever the API
# and the current model happen to default to, and GEMINI_MODEL points at the
# moving alias gemini-flash-latest. A module constant, not a setting - there is
# no per-deployment reason to vary it on a localhost gateway.
_SAFETY_THRESHOLD = types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
_SAFETY_SETTINGS = [
    types.SafetySetting(category=category, threshold=_SAFETY_THRESHOLD)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
    # HARM_CATEGORY_CIVIC_INTEGRITY is deliberately left unset: it is off
    # upstream by default and noisy on ordinary questions about politics.
]

# finish_reason is a taxonomy, not a flag. These mean the model refused to
# finish; MAX_TOKENS and STOP are emphatically not in here, because a truncated
# or simply empty reply is not a refusal.
# Which upstream statuses a different model could plausibly survive. Only 5xx:
# 429 is the shared account's quota and retrying compounds an exhaustion that is
# already happening; 403 is a revoked key and 400 a malformed request, identical
# on either tier; 404 is a mistyped model id, and retrying it would quietly hide
# the typo behind a working fallback forever.
def _is_retryable(code: int | None) -> bool:
    return code is not None and 500 <= code < 600


_OUTPUT_REFUSED = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
    }
)


class LLMError(RuntimeError):
    """Raised when the upstream model call cannot be completed.

    `retryable` says whether trying the other routing tier could plausibly do
    better. It is a required keyword rather than a defaulted one: a default
    would make every future raise site retryable by omission, which is the
    failure that fails open.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ContentBlocked(RuntimeError):
    """Raised when Gemini refused the caller's *input*.

    A sibling of `LLMError`, not a subclass, on purpose: the router maps this to
    400 and `LLMError` to 502, and if this inherited from `LLMError` that
    mapping would hold only while the `except` clauses stayed in the right
    order. As siblings, catch order cannot silently regress it.
    """


@lru_cache
def _get_client() -> genai.Client:
    settings = get_settings()
    if not settings.is_configured:
        # No request was ever sent, so the other tier would fail identically.
        raise LLMError(
            "GOOGLE_API_KEY is not set. Add it to your .env file.", retryable=False
        )
    return genai.Client(api_key=settings.google_api_key)


async def generate_reply(message: str, model: str) -> str:
    """Send `message` to `model` and return the model's text reply.

    The model is a parameter rather than a settings lookup so that the caller
    that chose it (app/routing.py) is also the one that can report which model
    actually answered.
    """
    client = _get_client()

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=message,
            # No tools are registered, so skip the SDK's function-calling loop.
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                safety_settings=_SAFETY_SETTINGS,
            ),
        )
    except genai_errors.APIError as exc:
        logger.warning("Gemini API error (%s, code %s): %s", model, exc.code, exc)
        raise LLMError("Gemini API error", retryable=_is_retryable(exc.code)) from exc

    # The prompt itself was refused: the caller's input, so the caller's fault.
    # prompt_feedback is absent on an ordinary response, so guard it.
    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        logger.warning("Gemini blocked the prompt (%s): %s", model, feedback.block_reason)
        raise ContentBlocked("Message rejected by the model's safety filters.")

    text = (response.text or "").strip()
    if text:
        return text

    # No text. Work out whether the model refused or simply produced nothing.
    # `candidates` can be None or an empty list, so never index it blindly.
    candidates = response.candidates or ()
    finish_reason = candidates[0].finish_reason if candidates else None
    if finish_reason in _OUTPUT_REFUSED:
        # The model's own output was withheld. That is not the caller's input
        # being rejected, so it stays a 502 rather than blaming them for it.
        logger.warning("Gemini withheld the reply (%s): %s", model, finish_reason)
        # Same reasoning as ContentBlocked: identical content against identical
        # safety settings is not going to fare better on the other tier.
        raise LLMError("The model did not return a usable reply.", retryable=False)
    raise LLMError("Gemini returned an empty response.", retryable=True)
