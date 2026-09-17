"""Routing over the wire: the reported model, and composition with what exists.

The tiering and fallback policy itself is covered in tests/test_routing.py. What
matters here is that the response says which model actually answered, and that a
fallback stays invisible to the rate limiter and the guardrails.
"""

import contextlib

import pytest
from fastapi.testclient import TestClient

from app import routing
from app.main import app
from app.services.llm_client import ContentBlocked, LLMError
from tests.conftest import FAKE_REPLY, PRIMARY_KEY, SECONDARY_KEY

CHEAP = "cheap-model"
BIG = "big-model"
THRESHOLD = 100  # tokens, i.e. 400 characters

SHORT = "a" * 4  # 1 token
LONG = "a" * (THRESHOLD * 4)  # exactly at the threshold, so it escalates


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class FakeReplies:
    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failures = failures or {}

    async def __call__(self, message: str, model: str) -> str:
        self.calls.append((message, model))
        error = self.failures.get(model)
        if error is not None:
            raise error
        return FAKE_REPLY

    @property
    def models(self) -> list[str]:
        return [model for _, model in self.calls]


@pytest.fixture
def routed_client(monkeypatch, set_gateway_keys):
    """A TestClient with both tiers configured and a model-keyed fake upstream."""
    stack = contextlib.ExitStack()

    def _make(failures=None, limit: int = 5, threshold: int = THRESHOLD):
        monkeypatch.setenv("GEMINI_MODEL", BIG)
        monkeypatch.setenv("GEMINI_CHEAP_MODEL", CHEAP)
        monkeypatch.setenv("ROUTING_TOKEN_THRESHOLD", str(threshold))
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", str(limit))
        set_gateway_keys(f"{PRIMARY_KEY},{SECONDARY_KEY}")
        fake = FakeReplies(failures)
        monkeypatch.setattr(routing, "generate_reply", fake)
        return stack.enter_context(TestClient(app)), fake

    yield _make
    stack.close()


def post(client, message: str, key: str = PRIMARY_KEY):
    return client.post("/chat", json={"message": message}, headers=bearer(key))


def retryable() -> LLMError:
    return LLMError("boom", retryable=True)


# --- The reported model is the one that answered ----------------------------------


@pytest.mark.parametrize(
    ("message", "failures", "expected_model", "expected_calls"),
    [
        (SHORT, None, CHEAP, [CHEAP]),
        (LONG, None, BIG, [BIG]),
        (SHORT, {CHEAP: "retryable"}, BIG, [CHEAP, BIG]),
        (LONG, {BIG: "retryable"}, CHEAP, [BIG, CHEAP]),
    ],
    ids=["short", "long", "short with fallback", "long with fallback"],
)
def test_response_names_the_model_that_actually_answered(
    routed_client, message, failures, expected_model, expected_calls
):
    prepared = {model: retryable() for model in (failures or {})}
    client, fake = routed_client(prepared)

    response = post(client, message)

    assert response.status_code == 200
    # Asserted against the body, never against settings: settings.gemini_model
    # is only one of the two tiers and was what this used to report always.
    assert response.json()["model"] == expected_model
    assert response.json()["reply"] == FAKE_REPLY
    assert fake.models == expected_calls


def test_health_reports_both_tiers(routed_client):
    client, _ = routed_client()

    body = client.get("/health").json()

    assert body["model"] == BIG
    assert body["cheap_model"] == CHEAP


# --- Composition: the fallback is invisible to the limiter and the guardrails ------


def test_a_falling_back_request_costs_exactly_one_rate_limit_slot(
    routed_client, fake_clock
):
    # Every request falls back, so each one makes two upstream calls.
    client, fake = routed_client({CHEAP: retryable()}, limit=5)

    for _ in range(5):
        assert post(client, SHORT).status_code == 200

    # Five client requests consumed five slots, not ten.
    assert post(client, SHORT).status_code == 429
    # ...and the upstream cost is the bounded 2x the config comment describes.
    assert len(fake.calls) == 10


def test_upstream_calls_are_bounded_at_twice_the_limit(routed_client, fake_clock):
    client, fake = routed_client({CHEAP: retryable(), BIG: retryable()}, limit=3)

    for _ in range(3):
        assert post(client, SHORT).status_code == 502
    assert post(client, SHORT).status_code == 429

    # Three admitted requests, two calls each, and nothing beyond that.
    assert len(fake.calls) == 6


def test_a_guardrail_rejection_still_makes_no_upstream_call(routed_client):
    client, fake = routed_client()

    response = post(client, "my email is bob@example.com")

    assert response.status_code == 400
    assert fake.calls == []


def test_the_fallback_does_not_re_run_guardrails(routed_client):
    # A message that passes guardrails once must not be re-screened on retry;
    # the proof is that both calls carry the identical message.
    client, fake = routed_client({CHEAP: retryable()})

    assert post(client, SHORT).status_code == 200

    assert [sent for sent, _ in fake.calls] == [SHORT, SHORT]


def test_auth_failure_still_precedes_any_routing(routed_client):
    client, fake = routed_client()

    response = client.post("/chat", json={"message": SHORT}, headers=bearer("nope"))

    assert response.status_code == 401
    assert fake.calls == []


# --- Failure mapping --------------------------------------------------------------


def test_both_tiers_failing_is_a_502_with_no_upstream_detail(routed_client):
    client, fake = routed_client(
        {CHEAP: LLMError("cheap exploded", retryable=True),
         BIG: LLMError("big exploded", retryable=True)}
    )

    response = post(client, SHORT)

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream model request failed."}
    for internal in ("cheap exploded", "big exploded", CHEAP, BIG):
        assert internal not in response.text
    assert fake.models == [CHEAP, BIG]


def test_the_missing_key_message_never_reaches_a_client(routed_client):
    # This one names an environment variable, which is nobody's business but the
    # operator's - and it is non-retryable, so it costs exactly one call.
    client, fake = routed_client(
        {CHEAP: LLMError(
            "GOOGLE_API_KEY is not set. Add it to your .env file.", retryable=False
        )}
    )

    response = post(client, SHORT)

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream model request failed."}
    assert "GOOGLE_API_KEY" not in response.text
    assert len(fake.calls) == 1


def test_content_blocked_is_a_400_after_exactly_one_call(routed_client):
    client, fake = routed_client(
        {CHEAP: ContentBlocked("Message rejected by the model's safety filters.")}
    )

    response = post(client, SHORT)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Message rejected by the model's safety filters."
    }
    assert fake.models == [CHEAP]
