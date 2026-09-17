"""Shared fixtures: a TestClient wired to test settings and a fake model call.

No test may reach the real Gemini API. ``generate_reply`` is replaced with a
fake for every test, and the real client factory is swapped for one that
fails loudly, so a mistake costs a test failure instead of quota.
"""

import pytest
from fastapi.testclient import TestClient

from app import rate_limit, routing
from app.config import Settings, get_settings
from app.main import app
from app.services import llm_client

# Both are at least MIN_API_KEY_LENGTH characters, from RFC 6750's token set.
PRIMARY_KEY = "primary-" + "a1b2c3d4" * 5
SECONDARY_KEY = "secondary-" + "e5f6g7h8" * 5

FAKE_REPLY = "fake reply from the test double"


class FakeGenerateReply:
    """Stands in for generate_reply, recording the message and model per call.

    `calls` stays a list of messages so every existing assertion still reads
    naturally; `models` records which model each call went to, which is what
    the routing tests need.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.models: list[str] = []

    async def __call__(self, message: str, model: str) -> str:
        self.calls.append(message)
        self.models.append(model)
        return FAKE_REPLY


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch) -> FakeGenerateReply:
    """Replace the Gemini call in every test; fail if the real client is built."""
    fake = FakeGenerateReply()
    # routing.py imports generate_reply by name and is the only caller, so this
    # is where it must be patched. Patching app.routers.chat would silently
    # intercept nothing now that the handler calls generate_with_fallback.
    monkeypatch.setattr(routing, "generate_reply", fake)

    def refuse_real_client():
        raise AssertionError("A test tried to create a real Gemini client.")

    monkeypatch.setattr(llm_client, "_get_client", refuse_real_client)
    return fake


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Buckets are module-level state; stop them leaking between tests."""
    rate_limit.reset()
    yield
    rate_limit.reset()


class FakeClock:
    """A stand-in for time.monotonic that only moves when a test says so."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def fake_clock(monkeypatch) -> FakeClock:
    """Let rate-limit tests advance time instead of sleeping through a window."""
    clock = FakeClock()
    monkeypatch.setattr(rate_limit, "_now", clock)
    return clock


@pytest.fixture
def set_gateway_keys(monkeypatch):
    """Point the app at Settings built from a GATEWAY_API_KEYS value.

    Pass None to leave the variable unset. Settings are built through the real
    constructor, so parsing (commas, blanks, minimum length) is exercised too,
    then injected with dependency_overrides.
    """

    def _set(raw_keys: str | None) -> Settings:
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key-never-used")
        if raw_keys is None:
            monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
        else:
            monkeypatch.setenv("GATEWAY_API_KEYS", raw_keys)
        settings = Settings()
        app.dependency_overrides[get_settings] = lambda: settings
        return settings

    yield _set
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture
def client(set_gateway_keys):
    """TestClient with both test keys configured, as during a key rotation."""
    set_gateway_keys(f"{PRIMARY_KEY},{SECONDARY_KEY}")
    with TestClient(app) as test_client:
        yield test_client
