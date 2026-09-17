"""Rejections must explain themselves without quoting the request back.

Two leaks are covered here. A 422 used to serialize Pydantic's `input` field,
which is the whole rejected message. A 502 used to carry the upstream error
string, which can quote parts of the request and name the model and project.
"""

import asyncio
import logging

import pytest
from google.genai import errors as genai_errors

from app import routing
from app.routers.chat import MAX_MESSAGE_LENGTH
from app.services import llm_client
from tests.conftest import PRIMARY_KEY

# Realistic things a caller might have in a prompt that must not come back.
CARD = "4111111111111111"
EMAIL = "bob@example.com"
SECRET_TEXT = f"my card is {CARD} and my email is {EMAIL}"


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def assert_explains_without_echoing(response, *, rejected: str) -> None:
    """A 422 that still names the failing field but quotes none of the payload."""
    assert response.status_code == 422
    body = response.text

    # Still useful: the caller learns which field failed and why.
    errors = response.json()["detail"]
    assert errors, "the 422 body must still say what was wrong"
    first = errors[0]
    assert "loc" in first and "msg" in first and "type" in first

    # ...but nothing that quotes the request.
    assert "input" not in first
    assert "ctx" not in first
    for secret in (CARD, EMAIL, rejected):
        if secret:
            assert secret not in body


# --- 422: the rejected value never comes back ------------------------------------


def test_oversize_message_is_not_echoed(client):
    # The case that made this a real leak: the cap turned an over-long prompt
    # into a ~16 KB response containing that prompt.
    oversize = SECRET_TEXT + "x" * MAX_MESSAGE_LENGTH

    response = client.post(
        "/chat", json={"message": oversize}, headers=bearer(PRIMARY_KEY)
    )

    assert_explains_without_echoing(response, rejected=oversize)
    # The response is now a constant-size complaint, not a mirror.
    assert len(response.text) < 500


def test_oversize_message_by_one_character_is_not_echoed(client):
    oversize = SECRET_TEXT.ljust(MAX_MESSAGE_LENGTH + 1, "x")

    response = client.post(
        "/chat", json={"message": oversize}, headers=bearer(PRIMARY_KEY)
    )

    assert_explains_without_echoing(response, rejected=oversize)


def test_wrong_type_message_is_not_echoed(client):
    response = client.post(
        "/chat", json={"message": {"nested": SECRET_TEXT}}, headers=bearer(PRIMARY_KEY)
    )

    assert_explains_without_echoing(response, rejected=SECRET_TEXT)


def test_empty_message_still_reports_the_reason(client):
    response = client.post("/chat", json={"message": ""}, headers=bearer(PRIMARY_KEY))

    assert_explains_without_echoing(response, rejected="")
    assert response.json()["detail"][0]["loc"] == ["body", "message"]


def test_invalid_json_body_is_not_echoed(client):
    # A JSON decode error carries the raw body along in `input` too.
    truncated = ('{"message": "' + SECRET_TEXT).encode()

    response = client.post(
        "/chat",
        content=truncated,
        headers={**bearer(PRIMARY_KEY), "Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert CARD not in response.text
    assert EMAIL not in response.text


def test_unknown_field_is_not_echoed(client, fake_llm):
    # Extra fields are ignored rather than rejected; the point is that the
    # accepted path does not reflect them either.
    response = client.post(
        "/chat",
        json={"message": "hello", "note": SECRET_TEXT},
        headers=bearer(PRIMARY_KEY),
    )

    assert response.status_code == 200
    assert SECRET_TEXT not in response.text


# --- 502: the upstream error text never comes back --------------------------------


class FakeModels:
    """Stands in for client.aio.models, failing the way the SDK fails."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate_content(self, **kwargs):
        raise self._error


class FakeClient:
    def __init__(self, error: Exception) -> None:
        self.aio = type("Aio", (), {"models": FakeModels(error)})()


@pytest.fixture
def upstream_error(monkeypatch):
    """Make the real generate_reply hit an APIError from a fake transport."""

    def _set(error: Exception):
        monkeypatch.setattr(llm_client, "_get_client", lambda: FakeClient(error))
        return error

    return _set


# A realistic INVALID_ARGUMENT body: it quotes the request back at us.
LEAKY_RESPONSE_JSON = {
    "error": {
        "code": 400,
        "status": "INVALID_ARGUMENT",
        "message": f"Invalid value at 'contents': {SECRET_TEXT}",
        "details": [{"project": "my-private-project-42", "model": "gemini-secret"}],
    }
}


def test_upstream_error_text_is_logged_not_returned(upstream_error, caplog):
    caplog.set_level(logging.WARNING)
    error = genai_errors.APIError(400, LEAKY_RESPONSE_JSON)
    upstream_error(error)
    # Guard against a future SDK that stops embedding the body in str(exc):
    # without this the leak assertions below would pass vacuously.
    assert SECRET_TEXT in str(error)

    with pytest.raises(llm_client.LLMError) as raised:
        # No asyncio plugin is installed, and this call needs no running loop.
        asyncio.run(llm_client.generate_reply("say hello", "gemini-flash-latest"))

    # The exception message is for operators and carries none of the upstream
    # text; the router replaces it with a fixed string before answering anyway.
    detail = str(raised.value)
    for secret in (CARD, EMAIL, SECRET_TEXT, "my-private-project-42", "gemini-secret"):
        assert secret not in detail

    # The operator still gets the real reason.
    assert "Gemini API error" in caplog.text
    assert SECRET_TEXT in caplog.text


def test_upstream_error_detail_reaches_the_client_as_502(
    client, upstream_error, monkeypatch
):
    # conftest fakes routing.generate_reply wholesale; put the real one back so
    # the router sees the LLMError that generate_reply actually raises. Patching
    # app.routers.chat here would be a no-op and the test would pass vacuously.
    monkeypatch.setattr(routing, "generate_reply", llm_client.generate_reply)
    upstream_error(genai_errors.APIError(400, LEAKY_RESPONSE_JSON))

    response = client.post(
        "/chat", json={"message": "say hello"}, headers=bearer(PRIMARY_KEY)
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream model request failed."}
    for secret in (CARD, EMAIL, "my-private-project-42", "gemini-secret"):
        assert secret not in response.text
