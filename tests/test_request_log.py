"""The per-request summary line: `log_request` in app/main.py.

One JSON line per /chat request however it ended, carrying who (key_id), what
(method, path), how it ended (status, latency) and which model answered. The
existing logger.warning() calls stay the diagnostics; this is the summary.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app import main, routing
from app.main import app
from app.services.llm_client import ContentBlocked, LLMError
from tests.conftest import PRIMARY_KEY, SECONDARY_KEY, FakeClock

CHEAP = "test-cheap-model"
BIG = "test-big-model"
THRESHOLD = 100  # tokens, i.e. 400 characters

SHORT = "Say hello in one sentence."
LONG = "a" * (THRESHOLD * 4)

PRIMARY_KEY_ID = hashlib.sha256(PRIMARY_KEY.encode()).hexdigest()[:16]
WRONG_KEY = "wrong-" + "z9y8x7w6" * 5

FIELDS = {"ts", "key_id", "method", "path", "status", "latency_ms", "model"}
KEY_ID_RE = re.compile(r"[0-9a-f]{16}")


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class FakeUpstream:
    """generate_reply keyed by model: a reply, or an exception to raise.

    `clock` and `takes` let a test say how long the call lasted without
    sleeping; the middleware reads the same clock.
    """

    def __init__(self) -> None:
        self.reply = "fake reply from the test double"
        self.failures: dict[str, Exception] = {}
        self.clock: FakeClock | None = None
        self.takes = 0.0

    async def __call__(self, message: str, model: str) -> str:
        if self.clock is not None:
            self.clock.advance(self.takes)
        error = self.failures.get(model)
        if error is not None:
            raise error
        return self.reply


@pytest.fixture
def upstream(monkeypatch) -> FakeUpstream:
    fake = FakeUpstream()
    monkeypatch.setattr(routing, "generate_reply", fake)
    return fake


@pytest.fixture
def make_client(monkeypatch, set_gateway_keys, upstream):
    """A TestClient with both tiers configured; `limit` sets the rate limit."""
    clients = []

    def _make(limit: int = 50) -> TestClient:
        monkeypatch.setenv("GEMINI_MODEL", BIG)
        monkeypatch.setenv("GEMINI_CHEAP_MODEL", CHEAP)
        monkeypatch.setenv("ROUTING_TOKEN_THRESHOLD", str(THRESHOLD))
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", str(limit))
        set_gateway_keys(f"{PRIMARY_KEY},{SECONDARY_KEY}")
        # Unhandled errors must come back as a 500 rather than raise, so the
        # test can see that the line was still written.
        test_client = TestClient(app, raise_server_exceptions=False)
        clients.append(test_client)
        return test_client

    yield _make
    for test_client in clients:
        test_client.close()


@pytest.fixture
def lines(caplog):
    """Every summary line written so far, parsed."""
    caplog.set_level(logging.INFO, logger="app.request_log")

    def _lines() -> list[dict]:
        return [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == "app.request_log"
        ]

    return _lines


def only_line(lines) -> dict:
    written = lines()
    assert len(written) == 1, written
    return written[0]


def assert_well_formed(line: dict) -> None:
    """Pin the values, not just the names: `path` built from the full URL would
    keep the same key set and still leak the query string."""
    assert set(line) == FIELDS
    assert line["path"] == "/chat"
    assert line["method"] in {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "OTHER"}
    assert line["key_id"] == "unauthenticated" or KEY_ID_RE.fullmatch(line["key_id"])
    assert line["model"] in {CHEAP, BIG, None}
    assert type(line["status"]) is int
    assert type(line["latency_ms"]) is float
    ts = datetime.fromisoformat(line["ts"])
    assert ts.utcoffset() == timedelta(0)


# --- Success ------------------------------------------------------------------------


def test_success_writes_one_well_formed_line(make_client, lines):
    response = make_client().post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY))
    assert response.status_code == 200

    line = only_line(lines)
    assert_well_formed(line)
    assert line["key_id"] == PRIMARY_KEY_ID
    assert line["method"] == "POST"
    assert line["status"] == 200
    assert line["model"] == CHEAP == response.json()["model"]


def test_a_long_message_logs_the_escalation_tier(make_client, lines):
    make_client().post("/chat", json={"message": LONG}, headers=bearer(PRIMARY_KEY))
    assert only_line(lines)["model"] == BIG


def test_a_fallback_logs_the_model_that_actually_answered(make_client, upstream, lines):
    upstream.failures[CHEAP] = LLMError("overloaded", retryable=True)
    response = make_client().post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY))
    assert response.status_code == 200
    assert only_line(lines)["model"] == BIG


def test_health_writes_no_line(make_client, lines):
    assert make_client().get("/health").status_code == 200
    assert lines() == []


# --- Every layer that can reject still writes the line ------------------------------


def _email_in_reply(upstream):
    upstream.reply = "Write to bob@example.com"


def _blocked(upstream):
    upstream.failures[CHEAP] = ContentBlocked("Message rejected by the model's safety filters.")


def _upstream_down(upstream):
    upstream.failures[CHEAP] = LLMError("403 PERMISSION_DENIED", retryable=False)


def _crash(upstream):
    upstream.failures[CHEAP] = RuntimeError("a bug, not an LLMError")


REJECTIONS = {
    # name: (request kwargs, setup, status, authenticated, model)
    "401 missing header": ({"json": {"message": SHORT}}, None, 401, False, None),
    "401 wrong key": ({"json": {"message": SHORT}, "headers": bearer(WRONG_KEY)}, None, 401, False, None),
    "400 input guardrail": (
        {"json": {"message": "Mail me at bob@example.com"}, "headers": bearer(PRIMARY_KEY)},
        None, 400, True, None,
    ),
    "400 content blocked": (
        {"json": {"message": SHORT}, "headers": bearer(PRIMARY_KEY)}, _blocked, 400, True, None,
    ),
    "502 upstream failure": (
        {"json": {"message": SHORT}, "headers": bearer(PRIMARY_KEY)}, _upstream_down, 502, True, None,
    ),
    # The reply was produced, so the tier that produced it is recorded.
    "502 output guardrail": (
        {"json": {"message": SHORT}, "headers": bearer(PRIMARY_KEY)}, _email_in_reply, 502, True, CHEAP,
    ),
    "422 empty message": ({"json": {"message": ""}, "headers": bearer(PRIMARY_KEY)}, None, 422, True, None),
    # The body is parsed before dependencies run, so auth never ran: no identity.
    "422 malformed json with a valid key": (
        {"content": b"{not json", "headers": {**bearer(PRIMARY_KEY), "Content-Type": "application/json"}},
        None, 422, False, None,
    ),
    "500 unhandled": ({"json": {"message": SHORT}, "headers": bearer(PRIMARY_KEY)}, _crash, 500, True, None),
}


@pytest.mark.parametrize(
    ("kwargs", "setup", "status", "authenticated", "model"),
    REJECTIONS.values(),
    ids=REJECTIONS.keys(),
)
def test_every_outcome_writes_one_line(
    make_client, upstream, lines, kwargs, setup, status, authenticated, model
):
    if setup is not None:
        setup(upstream)
    assert make_client().post("/chat", **kwargs).status_code == status

    line = only_line(lines)
    assert_well_formed(line)
    assert line["status"] == status
    assert line["key_id"] == (PRIMARY_KEY_ID if authenticated else "unauthenticated")
    assert line["model"] == model


def test_rate_limited_request_writes_a_line_with_the_real_key_id(make_client, fake_clock, lines):
    test_client = make_client(limit=1)
    assert test_client.post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY)).status_code == 200
    assert test_client.post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY)).status_code == 429

    first, second = lines()
    assert_well_formed(second)
    assert second["status"] == 429
    assert second["key_id"] == PRIMARY_KEY_ID
    assert second["model"] is None


def test_wrong_method_is_logged_as_405(make_client, lines):
    assert make_client().get("/chat").status_code == 405
    line = only_line(lines)
    assert_well_formed(line)
    assert (line["method"], line["status"], line["key_id"]) == ("GET", 405, "unauthenticated")


# --- Latency is measured -------------------------------------------------------------


def test_latency_is_measured_not_hardcoded(make_client, upstream, lines, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(main, "_now", clock)
    upstream.clock = clock
    test_client = make_client()

    upstream.takes = 0.25
    test_client.post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY))
    upstream.takes = 0.04
    test_client.post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY))

    assert [line["latency_ms"] for line in lines()] == [250.0, 40.0]


# --- Where it goes, and what never goes there ----------------------------------------


def test_the_line_goes_to_stdout(make_client, capsys):
    make_client().post("/chat", json={"message": SHORT}, headers=bearer(PRIMARY_KEY))
    out = capsys.readouterr().out
    assert f'"key_id": "{PRIMARY_KEY_ID}"' in out


def test_no_key_message_or_reply_ever_reaches_the_line(make_client, upstream, lines, caplog, capsys):
    message = "MESSAGE-SENTINEL-" + "m" * 20
    reply = "REPLY-SENTINEL-" + "r" * 20
    upstream.reply = reply
    test_client = make_client()
    query = {"api_key": PRIMARY_KEY}

    requests = [
        ({"json": {"message": message}, "headers": bearer(PRIMARY_KEY), "params": query}, 200),
        ({"json": {"message": message}, "headers": bearer(WRONG_KEY)}, 401),
        ({"json": {"message": message}, "headers": {"Authorization": f"Basic {PRIMARY_KEY}"}}, 401),
        ({"json": {"message": message + " bob@example.com"}, "headers": bearer(PRIMARY_KEY)}, 400),
        ({"content": message.encode(), "headers": {**bearer(PRIMARY_KEY), "Content-Type": "application/json"}}, 422),
    ]
    for kwargs, expected in requests:
        assert test_client.post("/chat", **kwargs).status_code == expected

    # A reply that trips the output guardrail, then one that crashes.
    upstream.reply = reply + " bob@example.com"
    assert test_client.post("/chat", json={"message": message}, headers=bearer(PRIMARY_KEY)).status_code == 502
    upstream.failures[CHEAP] = RuntimeError(f"{message} {reply} {PRIMARY_KEY}")
    assert test_client.post("/chat", json={"message": message}, headers=bearer(PRIMARY_KEY)).status_code == 500

    # Make sure the log path really ran, so the check below isn't vacuous.
    written = lines()
    assert len(written) == 7

    captured = capsys.readouterr()
    output = "\n".join(json.dumps(line) for line in written) + captured.out
    for secret in (PRIMARY_KEY, WRONG_KEY, message, reply, "api_key"):
        assert secret not in output
        # A partial key is a leak too.
        assert secret[: len(secret) // 2] not in output
