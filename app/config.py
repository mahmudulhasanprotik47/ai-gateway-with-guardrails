"""Central place where environment variables are loaded and validated."""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

# Read .env once, at import time. Real environment variables win over the file.
load_dotenv(override=False)

logger = logging.getLogger(__name__)

# Client API keys shorter than this are too weak and are ignored.
MIN_API_KEY_LENGTH = 32

# Longest prompt /chat will accept. Lives here rather than in the router so the
# routing threshold can be sanity-checked against it without a circular import;
# app/routers/chat.py re-exports it.
MAX_MESSAGE_LENGTH = 16_000

# Per-client limit on /chat. Gemini's free-tier flash RPM sits in the ~10-15
# range and moves, and GEMINI_MODEL defaults to the alias gemini-flash-latest,
# so the ceiling can shift without a code change.
#
# This is no longer a 1:1 ratio with upstream calls. app/routing.py retries once
# on the other tier when a call fails with a retryable error, so one admitted
# request costs one upstream call normally and two in the worst case: 5/minute
# is 5 upstream calls in steady state and up to 10 during an outage, which
# reaches the low end of the free-tier range rather than sitting under it.
# That worst case is bounded and rare by construction - only 5xx responses and
# empty replies are retried, while 429 (quota), 403 (revoked key), 404 (bad
# model id) and 400 are not, so a retry storm cannot compound an exhausted
# quota. Lower this to 3 if the 2x ceiling matters more than the headroom.
DEFAULT_RATE_LIMIT_REQUESTS = 5
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

# Complexity routing. Short messages go to the cheaper tier and escalate upward,
# which is the shape that saves money: the common case is cheap.
# gemini-flash-lite-latest was confirmed against models.list() on the configured
# key rather than guessed, and is an alias for the same reason GEMINI_MODEL is -
# it tracks forward without a code change.
DEFAULT_CHEAP_MODEL = "gemini-flash-lite-latest"
DEFAULT_ROUTING_TOKEN_THRESHOLD = 1000

# Input guardrails that run on /chat. Named individually because they differ
# sharply in precision: email and card detection are high-confidence, while
# phone and injection carry a real false-positive rate an operator may need to
# switch off on its own.
KNOWN_GUARDRAIL_CHECKS = ("email", "card", "phone", "injection")


def _parse_api_keys(raw: str) -> tuple[str, ...]:
    """Split a comma-separated key list, dropping blank and too-short entries.

    Warnings identify an entry by its position only, never by its value.
    """
    keys: list[str] = []
    for position, entry in enumerate(raw.split(","), start=1):
        key = entry.strip()
        if not key:
            continue
        if len(key) < MIN_API_KEY_LENGTH:
            logger.warning(
                "GATEWAY_API_KEYS entry #%d is shorter than %d characters; ignoring it.",
                position,
                MIN_API_KEY_LENGTH,
            )
            continue
        keys.append(key)

    if not keys:
        logger.warning(
            "No valid GATEWAY_API_KEYS configured; /chat will reject every request."
        )
    return tuple(keys)


def _parse_routing_threshold() -> int:
    """Token threshold at or above which a message escalates to the big model.

    Warns, without serving a different value, when the threshold is set so high
    that nothing could ever reach it: estimate_tokens is len // 4 and messages
    are capped at MAX_MESSAGE_LENGTH, so anything above that quarter sends 100%
    of traffic to the cheap tier. That is a legitimate way to pin every request
    cheap, and it is also what a fat-fingered value looks like, so it is said
    out loud rather than silently obeyed.
    """
    threshold = _parse_int(
        "ROUTING_TOKEN_THRESHOLD", DEFAULT_ROUTING_TOKEN_THRESHOLD, minimum=0
    )
    unreachable_above = MAX_MESSAGE_LENGTH // 4
    if threshold > unreachable_above:
        logger.warning(
            "ROUTING_TOKEN_THRESHOLD is above %d, the most any accepted message "
            "can estimate, so every request will use the cheap model.",
            unreachable_above,
        )
    return threshold


def _parse_guardrail_checks(raw: str | None) -> frozenset[str]:
    """Split the comma-separated check list, the way `_parse_api_keys` does.

    Unset means every check is on. Set but empty turns them all off, which is
    the same deliberate "shut it" spelling `RATE_LIMIT_REQUESTS=0` already uses.
    An unrecognised name is dropped with a warning rather than failing the
    boot, so a typo cannot take the service down; it does mean a misspelled
    check is silently not running, hence the warning.
    """
    if raw is None:
        return frozenset(KNOWN_GUARDRAIL_CHECKS)

    checks: set[str] = set()
    for position, entry in enumerate(raw.split(","), start=1):
        name = entry.strip().lower()
        if not name:
            continue
        if name not in KNOWN_GUARDRAIL_CHECKS:
            # By position, never by value: a line-glue mistake in .env could put
            # an API key in here, exactly as it once did for GATEWAY_API_KEYS.
            logger.warning(
                "GUARDRAIL_CHECKS entry #%d is not a known check; known checks "
                "are %s. Ignoring it.",
                position,
                ", ".join(KNOWN_GUARDRAIL_CHECKS),
            )
            continue
        checks.add(name)

    if not checks:
        logger.warning("No guardrail checks enabled; /chat input is unfiltered.")
    return frozenset(checks)


def _parse_int(name: str, default: int, minimum: int) -> int:
    """Read an integer environment variable, or warn and use `default`.

    `minimum` differs per setting: a request limit of 0 is a deliberate "reject
    everything" and is honoured, while a window of 0 seconds is meaningless.
    The offending value is never logged - a line-glue mistake in `.env` could
    put someone's API key in it.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not an integer; using the default of %d.", name, default)
        return default
    if value < minimum:
        logger.warning(
            "%s must be at least %d; using the default of %d.", name, minimum, default
        )
        return default
    return value


class Settings:
    """Application settings sourced from the environment."""

    def __init__(self) -> None:
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.app_name: str = os.getenv("APP_NAME", "AI Gateway")
        # Keys clients present as "Authorization: Bearer <key>" (not the Gemini key).
        self.api_keys: tuple[str, ...] = _parse_api_keys(
            os.getenv("GATEWAY_API_KEYS", "")
        )
        # Per authenticated client, not global. 0 shuts /chat to every client.
        self.rate_limit_requests: int = _parse_int(
            "RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS, minimum=0
        )
        self.rate_limit_window_seconds: int = _parse_int(
            "RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_LIMIT_WINDOW_SECONDS, minimum=1
        )
        # Unset means all checks on; set-but-empty means all off.
        self.guardrail_checks: frozenset[str] = _parse_guardrail_checks(
            os.getenv("GUARDRAIL_CHECKS")
        )
        # Complexity routing: below the threshold use the cheap model, at or
        # above it use gemini_model. Setting them to the same value disables the
        # fallback retry, which is the off switch.
        self.gemini_cheap_model: str = os.getenv(
            "GEMINI_CHEAP_MODEL", DEFAULT_CHEAP_MODEL
        )
        self.routing_token_threshold: int = _parse_routing_threshold()

    @property
    def is_configured(self) -> bool:
        return bool(self.google_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
