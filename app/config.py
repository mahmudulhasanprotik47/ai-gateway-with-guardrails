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

# Per-client limit on /chat. Gemini's free-tier flash RPM sits in the ~10-15
# range and moves, and GEMINI_MODEL defaults to the alias gemini-flash-latest,
# so the ceiling can shift without a code change. 5 per minute is half the low
# end: one client cannot reach the upstream limit even if the alias resolves to
# the most restrictive model, and a second client still fits.
DEFAULT_RATE_LIMIT_REQUESTS = 5
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60


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

    @property
    def is_configured(self) -> bool:
        return bool(self.google_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
