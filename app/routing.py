"""Pick a model for each message, and fall back once if that model fails.

Two small policies, deliberately kept out of `llm_client.py` so that module can
stay the thin transport wrapper its docstring claims to be.

**Tiering.** Short messages go to a cheaper model and escalate upward, so the
common case is the cheap one. "Short" is measured with a crude estimate, not a
token count - see `estimate_tokens`.

**Fallback.** A failed call is retried exactly once against the other tier, and
only when the failure is one a different model could plausibly survive. This is
tier fallback inside one vendor on one API key: it rides out a single model
being slow, overloaded or briefly unavailable. It does **not** survive Google
being down, the key being revoked, or the account's quota being spent, because
both tiers sit behind the same key and the same account. Real multi-provider
fallback would need a second vendor and a second paid key, which this project
has deliberately never required, and that is the honest boundary here.
"""

import logging

from app.config import Settings
from app.services.llm_client import LLMError, generate_reply

logger = logging.getLogger(__name__)


def estimate_tokens(message: str) -> int:
    """Roughly how many tokens `message` is, as characters divided by four.

    An estimate and nothing more. It is wrong for CJK, which packs far more
    tokens per character, wrong for code and punctuation-dense text, and wrong
    for any non-Latin script. It is used only to choose which model answers,
    never to bill, cap, or report anything, so being 30% out changes the tier
    and has no other consequence. `MAX_MESSAGE_LENGTH` already encodes the same
    four-characters-per-token assumption informally.

    The exact alternative, `client.models.count_tokens()`, is a network round
    trip against the very quota the cheap tier exists to conserve: an API call
    to decide which API call to make.
    """
    return len(message) // 4


def choose_model(message: str, settings: Settings) -> str:
    """Cheap model below the threshold, the configured model at or above it."""
    if estimate_tokens(message) < settings.routing_token_threshold:
        return settings.gemini_cheap_model
    return settings.gemini_model


def _same_model(left: str, right: str) -> bool:
    """Compare model ids ignoring the optional `models/` prefix the API uses."""
    return left.removeprefix("models/") == right.removeprefix("models/")


async def generate_with_fallback(
    message: str, settings: Settings, key_id: str
) -> tuple[str, str]:
    """Answer `message`, returning the reply and the model that produced it.

    Guardrails already ran on this message in the handler, and the retry reuses
    it unchanged, so nothing is re-checked. The rate limiter ran even earlier,
    as a router dependency, so a retry cannot cost the caller a second slot -
    one client request is one slot however many upstream calls it takes.
    """
    primary = choose_model(message, settings)
    alternate = (
        settings.gemini_model
        if primary == settings.gemini_cheap_model
        else settings.gemini_cheap_model
    )

    try:
        return await generate_reply(message, primary), primary
    except LLMError as exc:
        if not exc.retryable:
            raise
        if _same_model(alternate, primary):
            # Both tiers are the same model, which is how an operator turns the
            # fallback off. Retrying would just repeat the call that failed.
            raise
        # WARNING, with both models and the client, because this is invisible
        # otherwise: a mistyped cheap model would be masked by a fallback that
        # always works, and every short request would quietly cost two calls and
        # twice the latency. The key_id is what separates a real outage from one
        # client farming fallbacks.
        logger.warning(
            "Falling back from %s to %s for client %s: %s",
            primary,
            alternate,
            key_id,
            exc,
        )

    # Outside the except block on purpose: exactly one retry, with no loop and
    # no counter to tune into one, and a second failure that is not chained onto
    # the first.
    return await generate_reply(message, alternate), alternate
