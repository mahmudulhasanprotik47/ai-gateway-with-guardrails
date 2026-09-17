"""Safety settings and blocked-response handling (app/services/llm_client.py).

These call the real `generate_reply`, so they cannot go through the `/chat`
route: tests/conftest.py replaces `routing.generate_reply` wholesale. The transport
is faked one layer lower, at `_get_client`, so the code under test is the real
one and no request ever leaves the machine.
"""

import asyncio
import logging

import pytest
from google.genai import errors as genai_errors, types

from app.services import llm_client


class FakeModels:
    """Stands in for client.aio.models, capturing the config it was handed."""

    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _RaisingModels:
    """Like FakeModels, but the call fails instead of returning."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        raise self._error


class FakeClient:
    def __init__(self, response) -> None:
        self.models = FakeModels(response)
        self.aio = self

    def __call__(self):  # so it can be used directly as _get_client
        return self


@pytest.fixture
def fake_gemini(monkeypatch):
    """Point generate_reply at a fake transport returning `response`."""

    def _set(response) -> FakeModels:
        client = FakeClient(response)
        monkeypatch.setattr(llm_client, "_get_client", lambda: client)
        return client.models

    return _set


@pytest.fixture
def failing_gemini(monkeypatch):
    """Point generate_reply at a transport that raises `error`."""

    def _set(error: Exception) -> _RaisingModels:
        client = FakeClient(None)
        client.models = _RaisingModels(error)
        monkeypatch.setattr(llm_client, "_get_client", lambda: client)
        return client.models

    return _set


MODEL = "gemini-flash-latest"


def reply(message: str = "say hello", model: str = MODEL) -> str:
    # No asyncio plugin is installed and this needs no running loop.
    return asyncio.run(llm_client.generate_reply(message, model))


def make_response(
    *, text: str | None = None, finish_reason=None, block_reason=None, candidates=None
) -> types.GenerateContentResponse:
    """Build a response the way the SDK would hand one back."""
    if candidates is None:
        if text is None and finish_reason is None:
            candidates = []
        else:
            content = (
                types.Content(role="model", parts=[types.Part(text=text)])
                if text is not None
                else None
            )
            candidates = [types.Candidate(content=content, finish_reason=finish_reason)]

    feedback = (
        types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
        if block_reason is not None
        else None
    )
    return types.GenerateContentResponse(
        candidates=candidates, prompt_feedback=feedback
    )


# --- Safety settings are actually sent -------------------------------------------


def test_safety_settings_are_sent_for_the_four_categories(fake_gemini):
    models = fake_gemini(make_response(text="hi"))

    assert reply() == "hi"

    config = models.calls[0]["config"]
    sent = {setting.category: setting.threshold for setting in config.safety_settings}
    assert sent == {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
            types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
            types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
            types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
            types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    }


def test_civic_integrity_is_deliberately_left_unset(fake_gemini):
    models = fake_gemini(make_response(text="hi"))
    reply()

    categories = {s.category for s in models.calls[0]["config"].safety_settings}
    assert types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY not in categories


def test_the_requested_model_is_forwarded_to_the_api(fake_gemini):
    """The one kwarg the whole model-accuracy fix rests on."""
    models = fake_gemini(make_response(text="hi"))

    reply(model="gemini-flash-lite-latest")

    assert models.calls[0]["model"] == "gemini-flash-lite-latest"


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (500, True),
        (503, True),
        (429, False),
        (403, False),
        (404, False),
        (400, False),
    ],
    ids=["500", "503", "429 quota", "403 revoked", "404 bad model", "400 bad request"],
)
def test_api_error_retryability_follows_the_status_code(failing_gemini, code, retryable):
    """Only a 5xx could plausibly go better on the other tier.

    429 is the shared account's quota, so retrying compounds an exhaustion that
    is already happening; 403 and 400 fail identically either way; and retrying
    a 404 would hide a mistyped model id behind a fallback that always works.
    """
    failing_gemini(
        genai_errors.APIError(code, {"error": {"code": code, "message": "nope"}})
    )

    with pytest.raises(llm_client.LLMError) as raised:
        reply()

    assert raised.value.retryable is retryable


def test_automatic_function_calling_is_still_disabled(fake_gemini):
    # Guard against the safety settings quietly replacing the existing config.
    models = fake_gemini(make_response(text="hi"))
    reply()

    config = models.calls[0]["config"]
    assert config.automatic_function_calling.disable is True


# --- The prompt was blocked: the caller's input, so 400 ---------------------------


@pytest.mark.parametrize(
    "block_reason",
    [
        types.BlockedReason.SAFETY,
        types.BlockedReason.PROHIBITED_CONTENT,
        types.BlockedReason.BLOCKLIST,
    ],
    ids=["safety", "prohibited", "blocklist"],
)
def test_blocked_prompt_raises_content_blocked(fake_gemini, block_reason, caplog):
    caplog.set_level(logging.WARNING)
    fake_gemini(make_response(block_reason=block_reason))

    with pytest.raises(llm_client.ContentBlocked) as raised:
        reply()

    assert "safety filters" in str(raised.value)
    assert "Gemini blocked the prompt" in caplog.text


def test_content_blocked_is_not_an_llm_error():
    """Sibling, not subclass, so the router's catch order cannot matter."""
    assert not issubclass(llm_client.ContentBlocked, llm_client.LLMError)
    assert not issubclass(llm_client.LLMError, llm_client.ContentBlocked)


def test_blocked_prompt_wins_over_an_empty_candidate_list(fake_gemini):
    # Both conditions hold at once; the block is the more specific truth.
    fake_gemini(
        make_response(block_reason=types.BlockedReason.SAFETY, candidates=[])
    )

    with pytest.raises(llm_client.ContentBlocked):
        reply()


# --- The output was withheld: not the caller's fault, so 502 ----------------------


@pytest.mark.parametrize(
    "finish_reason",
    [
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
    ],
    ids=["safety", "recitation", "prohibited", "spii"],
)
def test_withheld_output_is_an_llm_error_not_content_blocked(
    fake_gemini, finish_reason, caplog
):
    caplog.set_level(logging.WARNING)
    fake_gemini(make_response(finish_reason=finish_reason))

    with pytest.raises(llm_client.LLMError) as raised:
        reply()

    assert "did not return a usable reply" in str(raised.value)
    assert "Gemini withheld the reply" in caplog.text


def test_max_tokens_with_no_text_is_not_treated_as_a_block(fake_gemini):
    # finish_reason is a taxonomy, not a flag: truncation is not a refusal.
    fake_gemini(make_response(finish_reason=types.FinishReason.MAX_TOKENS))

    with pytest.raises(llm_client.LLMError) as raised:
        reply()

    assert "empty response" in str(raised.value)


def test_max_tokens_with_text_still_returns_the_text(fake_gemini):
    fake_gemini(
        make_response(text="a partial ", finish_reason=types.FinishReason.MAX_TOKENS)
    )

    assert reply() == "a partial"


# --- Degenerate responses must not crash -------------------------------------------


def test_empty_candidate_list_does_not_raise_index_error(fake_gemini):
    fake_gemini(make_response(candidates=[]))

    with pytest.raises(llm_client.LLMError) as raised:
        reply()

    assert "empty response" in str(raised.value)


def test_missing_candidates_does_not_raise(fake_gemini):
    fake_gemini(make_response(candidates=None, text=None))

    with pytest.raises(llm_client.LLMError):
        reply()


def test_absent_prompt_feedback_is_fine(fake_gemini):
    # An ordinary successful response carries no prompt_feedback at all.
    models = fake_gemini(make_response(text="all good"))

    assert reply() == "all good"
    assert models.calls
