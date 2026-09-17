"""Model tiering and one-shot fallback (app/routing.py), at unit level.

The test double fails according to which model it is called with, so every case
below falls out of a policy rather than a call counter.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from app import routing
from app.config import (
    DEFAULT_CHEAP_MODEL,
    DEFAULT_ROUTING_TOKEN_THRESHOLD,
    MAX_MESSAGE_LENGTH,
    Settings,
)
from app.services.llm_client import ContentBlocked, LLMError

CHEAP = "cheap-model"
BIG = "big-model"
THRESHOLD = 100  # tokens, i.e. 400 characters
KEY_ID = "0123456789abcdef"

REPLY = "a reply"


def settings(cheap: str = CHEAP, big: str = BIG, threshold: int = THRESHOLD):
    """Only the four attributes routing reads."""
    return SimpleNamespace(
        gemini_cheap_model=cheap,
        gemini_model=big,
        routing_token_threshold=threshold,
    )


def message_of(tokens: int) -> str:
    """A message that estimates to exactly `tokens` under len // 4."""
    return "a" * (tokens * 4)


class FakeReplies:
    """Records every (message, model) call and fails per a model-keyed policy."""

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failures = failures or {}

    async def __call__(self, message: str, model: str) -> str:
        self.calls.append((message, model))
        error = self.failures.get(model)
        if error is not None:
            raise error
        return REPLY

    @property
    def models(self) -> list[str]:
        return [model for _, model in self.calls]


@pytest.fixture
def replies(monkeypatch):
    def _set(failures: dict[str, Exception] | None = None) -> FakeReplies:
        fake = FakeReplies(failures)
        monkeypatch.setattr(routing, "generate_reply", fake)
        return fake

    return _set


def run(message: str, config=None) -> tuple[str, str]:
    return asyncio.run(
        routing.generate_with_fallback(message, config or settings(), KEY_ID)
    )


def retryable(message: str = "boom") -> LLMError:
    return LLMError(message, retryable=True)


def permanent(message: str = "boom") -> LLMError:
    return LLMError(message, retryable=False)


# --- estimate_tokens -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("a", 0), ("aaa", 0), ("aaaa", 1), ("aaaaaaa", 1), ("aaaaaaaa", 2)],
)
def test_estimate_tokens_is_characters_over_four(text, expected):
    assert routing.estimate_tokens(text) == expected


def test_estimate_tokens_is_documented_as_an_estimate():
    # Not a real tokenizer: four ASCII characters and one CJK character are both
    # "one token" to this function, and the CJK figure is badly wrong. Pinned so
    # nobody mistakes it for exact.
    assert routing.estimate_tokens("中文测试") == 1


# --- choose_model, at the boundary -------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(0, CHEAP), (THRESHOLD - 1, CHEAP), (THRESHOLD, BIG), (THRESHOLD + 1, BIG)],
    ids=["empty", "just below", "exactly at", "just above"],
)
def test_choose_model_escalates_at_the_threshold(tokens, expected):
    # "Below the threshold" is cheap; "at or above" escalates.
    assert routing.choose_model(message_of(tokens), settings()) == expected


def test_a_threshold_of_zero_escalates_everything():
    config = settings(threshold=0)

    assert routing.choose_model("", config) == BIG
    assert routing.choose_model(message_of(1000), config) == BIG


def test_a_huge_threshold_keeps_even_the_longest_message_cheap():
    config = settings(threshold=MAX_MESSAGE_LENGTH)
    longest = "a" * MAX_MESSAGE_LENGTH

    assert routing.choose_model(longest, config) == CHEAP


def test_shipped_defaults_favour_the_cheap_tier():
    # The default threshold must sit well inside the accepted message size, or
    # the cheap tier would never be used (or always be used).
    assert 0 < DEFAULT_ROUTING_TOKEN_THRESHOLD < MAX_MESSAGE_LENGTH // 4
    assert DEFAULT_CHEAP_MODEL != Settings().gemini_model


# --- The happy path --------------------------------------------------------------


def test_short_message_uses_the_cheap_model(replies):
    fake = replies()

    reply, model = run(message_of(1))

    assert (reply, model) == (REPLY, CHEAP)
    assert fake.models == [CHEAP]


def test_long_message_uses_the_big_model(replies):
    fake = replies()

    reply, model = run(message_of(THRESHOLD))

    assert (reply, model) == (REPLY, BIG)
    assert fake.models == [BIG]


# --- Fallback --------------------------------------------------------------------


def test_retryable_failure_falls_back_up_to_the_big_model(replies, caplog):
    caplog.set_level(logging.WARNING)
    fake = replies({CHEAP: retryable()})

    reply, model = run(message_of(1))

    assert (reply, model) == (REPLY, BIG)
    assert fake.models == [CHEAP, BIG]
    # Both models and the client, so an outage is distinguishable from one
    # client farming fallbacks, and a mistyped model is visible in the log.
    assert f"Falling back from {CHEAP} to {BIG}" in caplog.text
    assert KEY_ID in caplog.text


def test_retryable_failure_falls_back_down_to_the_cheap_model(replies):
    # The mirror case: escalation is not the only direction.
    fake = replies({BIG: retryable()})

    reply, model = run(message_of(THRESHOLD))

    assert (reply, model) == (REPLY, CHEAP)
    assert fake.models == [BIG, CHEAP]


def test_the_retry_reuses_the_very_same_message(replies):
    fake = replies({CHEAP: retryable()})
    message = message_of(1)

    run(message)

    # Guardrails ran once on this message before routing; the retry must not be
    # a different message that never went through them.
    assert [sent for sent, _ in fake.calls] == [message, message]


def test_content_blocked_is_never_retried(replies):
    fake = replies({CHEAP: ContentBlocked("blocked")})

    with pytest.raises(ContentBlocked):
        run(message_of(1))

    # Identical content against identical safety settings: the other tier would
    # refuse it too, so retrying only costs a call.
    assert fake.models == [CHEAP]


def test_a_non_retryable_llm_error_is_never_retried(replies):
    fake = replies({CHEAP: permanent()})

    with pytest.raises(LLMError):
        run(message_of(1))

    assert fake.models == [CHEAP]


def test_fallback_is_capped_at_exactly_one_retry(replies):
    fake = replies({CHEAP: retryable("first"), BIG: retryable("second")})

    with pytest.raises(LLMError) as raised:
        run(message_of(1))

    # Two calls, never three: the retry lives outside the except block, so there
    # is no loop to run again.
    assert fake.models == [CHEAP, BIG]
    assert str(raised.value) == "second"


def test_identical_tiers_disable_the_fallback(replies):
    # Setting both models the same is how an operator turns the retry off.
    fake = replies({CHEAP: retryable()})

    with pytest.raises(LLMError):
        run(message_of(1), settings(cheap=CHEAP, big=CHEAP))

    assert fake.models == [CHEAP]


def test_the_models_prefix_does_not_defeat_the_same_model_guard(replies):
    # The API spells these "models/x"; an operator may configure either form.
    fake = replies({CHEAP: retryable()})

    with pytest.raises(LLMError):
        run(message_of(1), settings(cheap=CHEAP, big=f"models/{CHEAP}"))

    assert fake.models == [CHEAP]


# --- Config ----------------------------------------------------------------------


def test_threshold_above_the_reachable_maximum_warns(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("ROUTING_TOKEN_THRESHOLD", str(MAX_MESSAGE_LENGTH))

    config = Settings()

    # Obeyed, not overridden - but said out loud, because it means every single
    # request goes to the cheap tier and that is easy to do by accident.
    assert config.routing_token_threshold == MAX_MESSAGE_LENGTH
    assert "every request will use the cheap model" in caplog.text


def test_threshold_at_the_reachable_maximum_is_quiet(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("ROUTING_TOKEN_THRESHOLD", str(MAX_MESSAGE_LENGTH // 4))

    Settings()

    assert "every request will use the cheap model" not in caplog.text


def test_cheap_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GEMINI_CHEAP_MODEL", "something-else")

    assert Settings().gemini_cheap_model == "something-else"
