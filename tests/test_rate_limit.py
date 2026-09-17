"""Per-client rate limiting on /chat (app/rate_limit.py).

Every rejection must be a 429 carrying Retry-After, must never reach the model,
and must never affect another client. Time is faked, so nothing here sleeps.
"""

import contextlib
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import (
    DEFAULT_RATE_LIMIT_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    Settings,
)
from app import routing
from app.main import app
from app.routers.chat import MAX_MESSAGE_LENGTH
from app.services.llm_client import LLMError
from tests.conftest import FAKE_REPLY, PRIMARY_KEY, SECONDARY_KEY

CHAT_BODY = {"message": "Say hello"}
WRONG_KEY = "wrong-" + "z9y8x7w6" * 5

# Small numbers keep the arithmetic in each test readable; the shipped defaults
# are checked separately in test_defaults_are_five_per_minute.
LIMIT = 3
WINDOW = 60


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def assert_limited(response, expected_retry_after: int | None = None) -> None:
    assert response.status_code == 429
    retry_after = response.headers["retry-after"]
    assert retry_after.isdigit() and int(retry_after) >= 1
    if expected_retry_after is not None:
        assert int(retry_after) == expected_retry_after
    assert response.json()["detail"].startswith("Rate limit exceeded.")


@pytest.fixture
def chat_client(monkeypatch, set_gateway_keys):
    """Build a TestClient with the limit and window this test wants.

    Values are passed through as strings so a test can hand over garbage and
    check the fallback.
    """
    stack = contextlib.ExitStack()

    def _make(requests: object = LIMIT, window: object = WINDOW) -> TestClient:
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", str(requests))
        monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", str(window))
        set_gateway_keys(f"{PRIMARY_KEY},{SECONDARY_KEY}")
        return stack.enter_context(TestClient(app))

    yield _make
    stack.close()


def post(client, key: str = PRIMARY_KEY, body=CHAT_BODY):
    return client.post("/chat", json=body, headers=bearer(key))


# --- Under, at, and over the limit ---------------------------------------------


def test_requests_under_the_limit_are_allowed(chat_client, fake_clock, fake_llm):
    client = chat_client()

    for _ in range(LIMIT - 1):
        assert post(client).status_code == 200
    assert len(fake_llm.calls) == LIMIT - 1


def test_request_at_the_limit_is_allowed(chat_client, fake_clock, fake_llm):
    client = chat_client()

    for _ in range(LIMIT - 1):
        assert post(client).status_code == 200

    # The limit-th request is the last allowed one, not the first rejected one.
    final = post(client)
    assert final.status_code == 200
    assert final.json()["reply"] == FAKE_REPLY
    assert len(fake_llm.calls) == LIMIT


def test_request_over_the_limit_is_rejected(chat_client, fake_clock, fake_llm):
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client).status_code == 200

    assert_limited(post(client), expected_retry_after=WINDOW)
    # A limited request must never reach the model.
    assert len(fake_llm.calls) == LIMIT


def test_rejection_carries_no_www_authenticate(chat_client, fake_clock):
    client = chat_client()
    for _ in range(LIMIT):
        post(client)

    response = post(client)

    assert_limited(response)
    # 429 is not an auth failure; a challenge here would only invite a retry
    # with a different key.
    assert "www-authenticate" not in response.headers


# --- Window behaviour ----------------------------------------------------------


def test_window_reset_rearms_the_full_limit(chat_client, fake_clock, fake_llm):
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client).status_code == 200
    assert_limited(post(client))

    fake_clock.advance(WINDOW + 1)

    for _ in range(LIMIT):
        assert post(client).status_code == 200
    assert_limited(post(client))
    assert len(fake_llm.calls) == LIMIT * 2


def test_retry_after_shrinks_as_the_window_advances(chat_client, fake_clock):
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client).status_code == 200

    assert_limited(post(client), expected_retry_after=WINDOW)
    fake_clock.advance(20)
    assert_limited(post(client), expected_retry_after=WINDOW - 20)
    fake_clock.advance(39)
    assert_limited(post(client), expected_retry_after=1)


def test_window_slides_rather_than_resetting(chat_client, fake_clock):
    """Timestamps expire one at a time, so a boundary frees one slot, not all.

    A fixed-window implementation passes every test above but fails this one: it
    would zero the counter at the boundary and allow a full LIMIT burst.
    """
    client = chat_client()
    start = fake_clock.now
    for offset in range(LIMIT):  # one request per second: 0s, 1s, 2s
        fake_clock.now = start + offset
        assert post(client).status_code == 200

    # Far enough for the first request to age out, and nothing else.
    fake_clock.now = start + WINDOW

    assert post(client).status_code == 200  # the one freed slot
    assert_limited(post(client), expected_retry_after=1)  # the second expires in 1s


def test_sustained_load_still_drains_the_window(chat_client, fake_clock):
    """A client that keeps retrying must still get through when its window ends.

    This is the test that fails if a rejected request is appended to the bucket:
    every retry would push the window forward and the client would stay shut out
    forever, while Retry-After kept promising relief.
    """
    client = chat_client()
    start = fake_clock.now
    for offset in range(LIMIT):
        fake_clock.now = start + offset
        assert post(client).status_code == 200

    first_success_at = None
    for second in range(LIMIT, WINDOW * 2):
        fake_clock.now = start + second
        if post(client).status_code == 200:
            first_success_at = second
            break

    # The oldest request was at `start`, so it ages out exactly one window later.
    assert first_success_at == WINDOW


# --- Isolation between clients --------------------------------------------------


def test_one_client_being_limited_does_not_affect_another(
    chat_client, fake_clock, fake_llm
):
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client, PRIMARY_KEY).status_code == 200
    assert_limited(post(client, PRIMARY_KEY))

    # The second key has its own untouched bucket.
    for _ in range(LIMIT):
        assert post(client, SECONDARY_KEY).status_code == 200
    assert_limited(post(client, SECONDARY_KEY))

    # ...and limiting it does not release the first.
    assert_limited(post(client, PRIMARY_KEY))
    assert len(fake_llm.calls) == LIMIT * 2


# --- Ordering and scope ----------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({}, "Missing or malformed Authorization header"),
        (bearer(WRONG_KEY), "Invalid API key"),
        ({"Authorization": "garbage"}, "Missing or malformed Authorization header"),
    ],
    ids=["no header", "wrong key", "malformed header"],
)
def test_auth_failure_beats_rate_limit(chat_client, fake_clock, headers, detail):
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client).status_code == 200
    assert_limited(post(client))

    # 401 always wins: the limiter runs only once an identity exists.
    response = client.post("/chat", json=CHAT_BODY, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": detail}
    assert "retry-after" not in response.headers


def test_health_is_never_rate_limited(chat_client, fake_clock):
    client = chat_client()

    for _ in range(LIMIT * 3):
        response = client.get("/health")
        assert response.status_code == 200
        assert "retry-after" not in response.headers


def test_invalid_body_consumes_a_slot(chat_client, fake_clock, fake_llm):
    """Dependencies run before body validation, so a 422 costs the caller a slot.

    That is the behaviour we want: malformed requests must not be free.
    """
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client, body={"message": ""}).status_code == 422

    assert_limited(post(client))
    assert fake_llm.calls == []


def test_upstream_failure_consumes_a_slot(chat_client, fake_clock, monkeypatch):
    """A 502 means we did call Gemini, so it counts against the caller."""

    async def failing_reply(message: str, model: str) -> str:
        # Not retryable, so this is exactly one upstream call per request and
        # the slot accounting stays easy to read.
        raise LLMError("upstream is unhappy", retryable=False)

    monkeypatch.setattr(routing, "generate_reply", failing_reply)
    client = chat_client()

    for _ in range(LIMIT):
        assert post(client).status_code == 502

    assert_limited(post(client))


def test_limit_change_applies_to_a_warm_bucket(chat_client, fake_clock):
    client = chat_client(requests=LIMIT)
    for _ in range(LIMIT):
        assert post(client).status_code == 200
    assert_limited(post(client))

    # Raising the limit releases the client immediately; the timestamps already
    # in the bucket still count against the new, larger allowance.
    client = chat_client(requests=LIMIT + 2)

    for _ in range(2):
        assert post(client).status_code == 200
    assert_limited(post(client))


# --- Configuration ---------------------------------------------------------------


def test_defaults_are_five_per_minute(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_REQUESTS", raising=False)
    monkeypatch.delenv("RATE_LIMIT_WINDOW_SECONDS", raising=False)

    settings = Settings()

    assert settings.rate_limit_requests == DEFAULT_RATE_LIMIT_REQUESTS == 5
    assert settings.rate_limit_window_seconds == DEFAULT_RATE_LIMIT_WINDOW_SECONDS == 60


def test_zero_requests_closes_chat_to_everyone(chat_client, fake_clock, fake_llm):
    """0 is honoured as a deliberate shutdown, not treated as garbage."""
    client = chat_client(requests=0)

    for key in (PRIMARY_KEY, SECONDARY_KEY):
        assert_limited(post(client, key), expected_retry_after=WINDOW)
    assert fake_llm.calls == []


BAD_LIMITS = {
    "not a number": "abc",
    "negative": "-1",
    "float": "2.5",
    "empty": "",
    "whitespace": "   ",
}


@pytest.mark.parametrize("raw", BAD_LIMITS.values(), ids=BAD_LIMITS.keys())
def test_bad_request_limit_falls_back_to_the_default(
    chat_client, fake_clock, fake_llm, raw
):
    client = chat_client(requests=raw)

    for _ in range(DEFAULT_RATE_LIMIT_REQUESTS):
        assert post(client).status_code == 200
    assert_limited(post(client))


@pytest.mark.parametrize(
    "raw", ["0", "-5", "nonsense"], ids=["zero", "negative", "text"]
)
def test_bad_window_falls_back_to_the_default(monkeypatch, raw):
    # A window of 0 is meaningless, unlike a request limit of 0.
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", raw)

    assert Settings().rate_limit_window_seconds == DEFAULT_RATE_LIMIT_WINDOW_SECONDS


def test_bad_config_warns_without_echoing_the_value(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    # A line-glue mistake in .env could park a real key in this variable, so the
    # warning must name the setting and never the value.
    glued = "5GATEWAY_API_KEYS=" + PRIMARY_KEY
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", glued)

    settings = Settings()

    assert settings.rate_limit_requests == DEFAULT_RATE_LIMIT_REQUESTS
    assert "RATE_LIMIT_REQUESTS is not an integer" in caplog.text
    assert PRIMARY_KEY not in caplog.text
    assert glued not in caplog.text


# --- Prompt size -----------------------------------------------------------------
#
# The limiter counts requests, but Gemini bills tokens, so a cap on the prompt is
# what stops a single allowed request from costing whatever it likes.


def test_message_at_the_size_limit_is_accepted(chat_client, fake_clock, fake_llm):
    client = chat_client()

    response = post(client, body={"message": "a" * MAX_MESSAGE_LENGTH})

    assert response.status_code == 200
    assert len(fake_llm.calls) == 1


def test_oversized_message_is_rejected_before_the_model(
    chat_client, fake_clock, fake_llm
):
    client = chat_client()

    response = post(client, body={"message": "a" * (MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 422
    assert fake_llm.calls == []


# --- Logging ------------------------------------------------------------------------


def test_429_leaks_neither_the_key_nor_the_key_id(
    chat_client, fake_clock, caplog, capsys
):
    caplog.set_level(logging.DEBUG)
    client = chat_client()
    for _ in range(LIMIT):
        assert post(client).status_code == 200

    response = post(client)
    assert_limited(response)
    body = response.text
    output = caplog.text + capsys.readouterr().out

    # Make sure the log path really ran, so the checks below are not vacuous.
    assert "Rate limited /chat for client" in output
    # key_id is a non-secret label: fine in our logs, not in the client's body.
    key_id = output.split("for client ")[1].split(":")[0].strip()
    assert key_id and key_id not in body
    for secret in (PRIMARY_KEY, SECONDARY_KEY):
        assert secret not in output and secret not in body
        assert secret[: len(secret) // 2] not in output
