"""Output guardrails on /chat: `check_reply` in app/guardrails.py.

Same split as the input side. The corpus runs against `check_reply` directly,
because /chat allows five requests a minute; HTTP is used only where the
question is about the wire: the 502, identical bodies, no fallback, the rate
limit, and leakage into the response or the log.

The detectors themselves are covered in tests/test_guardrails.py, so PII gets
one case per category here, not a second corpus.
"""

import base64
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.config import KNOWN_GUARDRAIL_CHECKS
from app.guardrails import _REPLY_WITHHELD_DETAIL, check_message, check_reply
from app.routers import chat
from tests.conftest import FAKE_REPLY, GOOGLE_KEY, PRIMARY_KEY, SECONDARY_KEY

KEY_ID = "0123456789abcdef"

EMAIL = "bob@example.com"
CARD = "4111111111111111"
PHONE = "(555) 010-1234"


def settings(checks=KNOWN_GUARDRAIL_CHECKS, google_api_key=GOOGLE_KEY):
    return SimpleNamespace(
        guardrail_checks=frozenset(checks),
        api_keys=(PRIMARY_KEY, SECONDARY_KEY),
        google_api_key=google_api_key,
    )


def check(reply: str, config=None) -> None:
    check_reply(reply, config or settings(), KEY_ID)


def withhold(reply: str, config=None) -> HTTPException:
    """Assert the reply is withheld and hand back the 502 for inspection."""
    with pytest.raises(HTTPException) as raised:
        check(reply, config)
    assert raised.value.status_code == 502
    assert raised.value.detail == _REPLY_WITHHELD_DETAIL
    return raised.value


def fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


# --- Clean replies pass ------------------------------------------------------------

CLEAN = {
    "prose": "Hello! Here is a one-sentence greeting for you.",
    "code fence": "```python\ndef add(a, b):\n    return a + b\n```",
    "list": "1. Measure\n2. Cut\n3. Measure again",
    "year and money": "In 2024 the budget was $1,500,000.",
}


@pytest.mark.parametrize("reply", CLEAN.values(), ids=CLEAN.keys())
def test_clean_reply_passes(reply):
    check(reply)  # must not raise


# --- False positives that must NOT trip ----------------------------------------------

MUST_NOT_TRIP = {
    "talks about api keys": "Store API keys in environment variables, never in code.",
    "luhn-failing 16 digits": "Your order number is 4111111111111112.",
    # Whole-value matching: one character short of a key is not the key.
    "prefix of a gateway key": f"Here you go: {PRIMARY_KEY[:-1]}",
    "some other 40-char token": "Token: " + "q9r8s7t6" * 5,
    # The injection heuristic judges requests, not answers.
    "explains an injection": "Attackers write 'ignore all previous instructions'.",
    "discusses system prompts": "A system prompt sets the model's behaviour.",
}


@pytest.mark.parametrize("reply", MUST_NOT_TRIP.values(), ids=MUST_NOT_TRIP.keys())
def test_legitimate_reply_is_not_a_false_positive(reply):
    check(reply)  # must not raise


# --- Known false positives, accepted ---------------------------------------------------

KNOWN_FALSE_POSITIVES = {
    "example address in a regex answer": "This regex matches user@example.com.",
    "stripe test card": "Use the test card 4242 4242 4242 4242 in sandbox mode.",
    "555 example number": "Format it like 555-123-4567.",
}


@pytest.mark.parametrize(
    "reply", KNOWN_FALSE_POSITIVES.values(), ids=KNOWN_FALSE_POSITIVES.keys()
)
def test_pinned_known_false_positives(reply):
    """Example data a model emits routinely, withheld all the same.

    Accepted on purpose rather than allowlisted: the detectors cannot tell a
    documentation address or a sandbox card from a real one. Pinned so the cost
    stays visible, and so an allowlist, if one arrives, is a decision.
    """
    withhold(reply)


# --- PII: one per category ---------------------------------------------------------

PII = {
    "email": f"You can reach them at {EMAIL}.",
    "card": f"The card on file is {CARD}.",
    "card spaced": "The card on file is 4111 1111 1111 1111.",
    "phone": f"Call {PHONE} tomorrow.",
}


@pytest.mark.parametrize("reply", PII.values(), ids=PII.keys())
def test_pii_in_reply_is_withheld(reply):
    withhold(reply)


# --- Secrets --------------------------------------------------------------------------

SECRETS = {
    "primary gateway key": PRIMARY_KEY,
    "secondary gateway key (rotation)": SECONDARY_KEY,
    "google api key": GOOGLE_KEY,
    "mid-sentence": f"Sure, the key is {PRIMARY_KEY} as requested.",
    "in a code fence": f"```\nAUTH={PRIMARY_KEY}\n```",
    "fullwidth": fullwidth(PRIMARY_KEY),
    "split by a zero-width space": PRIMARY_KEY[:20] + "​" + PRIMARY_KEY[20:],
}


@pytest.mark.parametrize("reply", SECRETS.values(), ids=SECRETS.keys())
def test_secret_in_reply_is_withheld(reply):
    withhold(reply)


BYPASSES = {
    "key spaced out": " ".join(PRIMARY_KEY),
    "key base64": base64.b64encode(PRIMARY_KEY.encode()).decode(),
    "card in words": "four one one one, one one one one, one one one one, one one one one",
    "card one digit per line": "\n".join(CARD),
}


@pytest.mark.parametrize("reply", BYPASSES.values(), ids=BYPASSES.keys())
def test_documented_output_bypasses_are_not_caught(reply):
    """A caller who wants content out asks for it in another shape.

    Pinned so check_reply's "tripwire, not a boundary" docstring stays honest:
    if one of these starts being caught, that wording deserves another look.
    """
    check(reply)  # must not raise: documented as out of reach


# --- Configuration --------------------------------------------------------------------


def test_empty_google_key_does_not_block_every_reply():
    # "" is a substring of everything; an unset key must not withhold it all.
    check("An ordinary reply.", settings(google_api_key=""))


def test_empty_guardrail_checks_disables_output_pii_but_not_the_secret_check():
    config = settings(checks=())

    check(f"{EMAIL}, {CARD}, {PHONE}", config)  # PII is off
    withhold(f"key: {PRIMARY_KEY}", config)  # the secret check has no switch


def test_phone_turned_off_is_off_on_both_sides():
    config = settings(checks=("email", "card", "injection"))
    reply = f"Call {PHONE} tomorrow."

    check_message(reply, config, KEY_ID)  # input: must not raise
    check(reply, config)  # output: must not raise


# --- The rejection itself -----------------------------------------------------------

TRIGGERS = {
    "email": EMAIL,
    "card": CARD,
    "phone": PHONE,
    "primary key": PRIMARY_KEY,
    "secondary key": SECONDARY_KEY,
    "google key": GOOGLE_KEY,
    "everything at once": f"{EMAIL} {CARD} {PHONE} {PRIMARY_KEY} {GOOGLE_KEY}",
}


@pytest.mark.parametrize("trigger", TRIGGERS.values(), ids=TRIGGERS.keys())
def test_every_trigger_raises_the_identical_exception(trigger):
    error = withhold(f"Here it is: {trigger}")

    # Nothing derived from the reply, and nothing saying which check fired.
    assert error.detail == "Reply withheld by the gateway."
    assert trigger not in error.detail
    assert error.headers is None
    # A fresh exception: no upstream error or earlier failure rides along.
    assert error.__cause__ is None
    assert error.__context__ is None


def test_withheld_detail_is_distinct_from_the_upstream_failure():
    assert _REPLY_WITHHELD_DETAIL != chat._UPSTREAM_FAILURE_DETAIL


def test_log_names_reason_and_categories_but_no_content(caplog):
    caplog.set_level(logging.DEBUG)
    reply = f"card {CARD}, keys {PRIMARY_KEY} {SECONDARY_KEY} {GOOGLE_KEY}"

    withhold(reply)

    # Non-vacuity: prove the log line fired, and that it is the output one.
    assert "Output guardrail (output_guardrail) withheld a reply" in caplog.text
    assert KEY_ID in caplog.text
    assert "card" in caplog.text and "secret" in caplog.text
    assert "Guardrail rejected a request" not in caplog.text

    for leaked in (reply, CARD, PRIMARY_KEY, SECONDARY_KEY, GOOGLE_KEY):
        assert leaked not in caplog.text
    # Partial values are leaks too.
    for secret in (PRIMARY_KEY, SECONDARY_KEY, GOOGLE_KEY):
        assert secret[: len(secret) // 2] not in caplog.text
    assert CARD[:8] not in caplog.text


# --- Over the wire --------------------------------------------------------------------


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def post(client, key: str = PRIMARY_KEY):
    return client.post("/chat", json={"message": "Say hello."}, headers=bearer(key))


def test_clean_reply_is_still_200(client, fake_llm):
    response = post(client)

    assert response.status_code == 200
    assert response.json()["reply"] == FAKE_REPLY


@pytest.mark.parametrize("trigger", TRIGGERS.values(), ids=TRIGGERS.keys())
def test_dirty_reply_is_502_with_the_fixed_body(client, fake_llm, trigger):
    fake_llm.reply = f"Here it is: {trigger}"

    response = post(client)

    assert response.status_code == 502
    assert response.json() == {"detail": "Reply withheld by the gateway."}
    assert trigger not in response.text
    assert "www-authenticate" not in response.headers
    assert "retry-after" not in response.headers
    # Checked after routing returns, so a block never reaches the fallback.
    assert len(fake_llm.calls) == 1


def test_body_is_identical_whichever_check_tripped(client, fake_llm):
    bodies = set()
    # Six requests, split across both keys to stay under the per-key limit.
    for index, trigger in enumerate(list(TRIGGERS.values())[:6]):
        fake_llm.reply = f"Here it is: {trigger}"
        response = post(client, PRIMARY_KEY if index % 2 else SECONDARY_KEY)
        assert response.status_code == 502
        bodies.add(response.content)

    assert len(bodies) == 1


def test_withheld_reply_leaks_into_neither_response_nor_log(client, fake_llm, caplog):
    caplog.set_level(logging.DEBUG)
    fake_llm.reply = f"card {CARD}, email {EMAIL}, key {PRIMARY_KEY}, {GOOGLE_KEY}"

    response = post(client)

    assert response.status_code == 502
    assert "Output guardrail (output_guardrail) withheld a reply" in caplog.text

    for leaked in (fake_llm.reply, CARD, EMAIL, PRIMARY_KEY, GOOGLE_KEY):
        assert leaked not in response.text
        assert leaked not in caplog.text
    for partial in (CARD[:8], "bob", PRIMARY_KEY[:16], GOOGLE_KEY[:13]):
        assert partial not in response.text
        assert partial not in caplog.text


def test_withheld_reply_consumes_a_rate_limit_slot(client, fake_clock, fake_llm):
    fake_llm.reply = f"The card is {CARD}."
    for _ in range(5):
        assert post(client).status_code == 502

    fake_llm.reply = FAKE_REPLY
    assert post(client).status_code == 429
