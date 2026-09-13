"""Central place where environment variables are loaded and validated."""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Read .env once, at import time. Real environment variables win over the file.
load_dotenv(override=False)


class Settings:
    """Application settings sourced from the environment."""

    def __init__(self) -> None:
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.app_name: str = os.getenv("APP_NAME", "AI Gateway")

    @property
    def is_configured(self) -> bool:
        return bool(self.google_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
