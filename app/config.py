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

    @property
    def is_configured(self) -> bool:
        return bool(self.google_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
