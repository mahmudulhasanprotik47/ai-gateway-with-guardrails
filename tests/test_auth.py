"""API key auth on /chat (app/auth.py).

Every rejection must be a 401 with ``WWW-Authenticate: Bearer`` and must never
reach the model.
"""

import logging

import pytest

from tests.conftest import FAKE_REPLY, PRIMARY_KEY, SECONDARY_KEY

CHAT_BODY = {"message": "Say hello"}
WRONG_KEY = "wrong-" + "z9y8x7w6" * 5

MALFORMED = "Missing or malformed Authorization header"
INVALID = "Invalid API key"


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def assert_rejected(response, detail: str) -> None:
    assert response.status_code == 401
    assert response.json() == {"detail": detail}
    assert response.headers["www-authenticate"] == "Bearer"


# --- Accepted keys -----------------------------------------------------------


def test_valid_key_is_accepted(client, fake_llm):
    response = client.post("/chat", json=CHAT_BODY, headers=bearer(PRIMARY_KEY))

    assert response.status_code == 200
    assert response.json()["reply"] == FAKE_REPLY
    assert fake_llm.calls == ["Say hello"]


def test_second_key_is_accepted_during_rotation(client, fake_llm):
    response = client.post("/chat", json=CHAT_BODY, headers=bearer(SECONDARY_KEY))

    assert response.status_code == 200
    assert response.json()["reply"] == FAKE_REPLY
    assert fake_llm.calls == ["Say hello"]


def test_retired_key_is_rejected_after_rotation(client, set_gateway_keys, fake_llm):
    set_gateway_keys(SECONDARY_KEY)

    assert_rejected(
        client.post("/chat", json=CHAT_BODY, headers=bearer(PRIMARY_KEY)), INVALID
    )
    ok = client.post("/chat", json=CHAT_BODY, headers=bearer(SECONDARY_KEY))
    assert ok.status_code == 200
    assert fake_llm.calls == ["Say hello"]


@pytest.mark.parametrize(
    "header",
    [
        f"bearer {PRIMARY_KEY}",
        f"BEARER {PRIMARY_KEY}",
        f"Bearer   {PRIMARY_KEY}",
        f"  Bearer {PRIMARY_KEY}  ",
    ],
    ids=["lowercase scheme", "uppercase scheme", "several spaces", "outer whitespace"],
)
def test_tolerated_header_formatting_is_accepted(client, header):
    response = client.post("/chat", json=CHAT_BODY, headers={"Authorization": header})

    assert response.status_code == 200


# --- Missing and malformed headers ---------------------------------------------


def test_missing_header_is_rejected(client, fake_llm):
    assert_rejected(client.post("/chat", json=CHAT_BODY), MALFORMED)
    assert fake_llm.calls == []


MALFORMED_HEADERS = {
    "empty": "",
    "whitespace only": "   ",
    "scheme only": "Bearer",
    "scheme and trailing space": "Bearer ",
    "no scheme": PRIMARY_KEY,
    "basic scheme": f"Basic {PRIMARY_KEY}",
    "token scheme": f"Token {PRIMARY_KEY}",
    "scheme repeated": f"Bearer Bearer {PRIMARY_KEY}",
    "colon after scheme": f"Bearer: {PRIMARY_KEY}",
    "no space after scheme": f"Bearer{PRIMARY_KEY}",
    "tab separator": f"Bearer\t{PRIMARY_KEY}",
    "trailing extra word": f"Bearer {PRIMARY_KEY} extra",
    "two tokens": f"Bearer {PRIMARY_KEY} {SECONDARY_KEY}",
    "comma-joined keys": f"Bearer {PRIMARY_KEY},{SECONDARY_KEY}",
    "quoted token": f'Bearer "{PRIMARY_KEY}"',
    "invalid character": f"Bearer {PRIMARY_KEY}!",
    "token over 512 chars": "Bearer " + "a" * 513,
}


@pytest.mark.parametrize(
    "header", MALFORMED_HEADERS.values(), ids=MALFORMED_HEADERS.keys()
)
def test_malformed_header_is_rejected(client, fake_llm, header):
    response = client.post("/chat", json=CHAT_BODY, headers={"Authorization": header})

    assert_rejected(response, MALFORMED)
    assert fake_llm.calls == []


def test_multiple_authorization_headers_are_rejected(client, fake_llm):
    # Both keys are valid on their own; sent together they are ambiguous.
    headers = [
        ("Authorization", f"Bearer {PRIMARY_KEY}"),
        ("Authorization", f"Bearer {SECONDARY_KEY}"),
    ]
    response = client.post("/chat", json=CHAT_BODY, headers=headers)

    assert_rejected(response, MALFORMED)
    assert fake_llm.calls == []


def test_token_of_exactly_512_chars_is_checked_as_a_key(client, fake_llm):
    # At the length limit the token is well-formed, so it fails as unknown.
    response = client.post("/chat", json=CHAT_BODY, headers=bearer("a" * 512))

    assert_rejected(response, INVALID)
    assert fake_llm.calls == []


# --- Wrong keys ----------------------------------------------------------------


def test_wrong_key_is_rejected(client, fake_llm):
    response = client.post("/chat", json=CHAT_BODY, headers=bearer(WRONG_KEY))

    assert_rejected(response, INVALID)
    assert fake_llm.calls == []


NEAR_MISS_KEYS = {
    "last character changed": PRIMARY_KEY[:-1] + "X",
    "first character changed": "X" + PRIMARY_KEY[1:],
    "one character short": PRIMARY_KEY[:-1],
    "one character extra": PRIMARY_KEY + "a",
    "different case": PRIMARY_KEY.upper(),
    "padding appended": PRIMARY_KEY + "=",
    "both keys concatenated": PRIMARY_KEY + SECONDARY_KEY,
}


@pytest.mark.parametrize("key", NEAR_MISS_KEYS.values(), ids=NEAR_MISS_KEYS.keys())
def test_near_miss_key_is_rejected(client, fake_llm, key):
    response = client.post("/chat", json=CHAT_BODY, headers=bearer(key))

    assert_rejected(response, INVALID)
    assert fake_llm.calls == []


@pytest.mark.parametrize(
    "raw_keys",
    [None, "", "   ", ", ,", "too-short-to-count"],
    ids=["unset", "empty", "whitespace", "only commas", "only a too-short key"],
)
def test_no_keys_configured_rejects_every_key(
    client, set_gateway_keys, fake_llm, raw_keys
):
    settings = set_gateway_keys(raw_keys)
    assert settings.api_keys == ()

    for key in (PRIMARY_KEY, SECONDARY_KEY, "too-short-to-count"):
        assert_rejected(
            client.post("/chat", json=CHAT_BODY, headers=bearer(key)), INVALID
        )
    assert fake_llm.calls == []


# --- Ordering and scope -------------------------------------------------------

INVALID_BODIES = {
    "empty message": {"message": ""},
    "missing message": {},
    "wrong type": {"message": 123},
    "no body": None,
}


@pytest.mark.parametrize(
    "body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys()
)
@pytest.mark.parametrize(
    ("headers", "detail"),
    [({}, MALFORMED), (bearer(WRONG_KEY), INVALID)],
    ids=["no header", "wrong key"],
)
def test_auth_is_checked_before_body_validation(
    client, fake_llm, body, headers, detail
):
    # Unauthenticated callers get a 401, never a 422 that describes the schema.
    response = client.post("/chat", json=body, headers=headers)

    assert_rejected(response, detail)
    assert fake_llm.calls == []


@pytest.mark.parametrize(
    "body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys()
)
def test_invalid_body_with_valid_key_is_422(client, fake_llm, body):
    # Control for the test above: these bodies really are invalid.
    response = client.post("/chat", json=body, headers=bearer(PRIMARY_KEY))

    assert response.status_code == 422
    assert fake_llm.calls == []


# Known gap: FastAPI parses the JSON body before it resolves dependencies, so a
# body that isn't valid JSON fails with 422 before require_api_key runs,
# whatever the Authorization header says. Well-formed JSON with the wrong shape
# is fine (see test_auth_is_checked_before_body_validation): those errors are
# collected and only raised after the dependencies. The 422 leaks only the
# JSON decode error, not the schema, and the model is never reached, so this is
# documented rather than fixed. strict=True makes the suite fail if the gap
# closes, so this marker gets removed then. raises=AssertionError means only
# the 401 assertion failing counts as the expected failure; any other error
# still fails the test.
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Invalid JSON returns 422 before the auth dependency runs",
)
@pytest.mark.parametrize(
    ("headers", "detail"),
    [({}, MALFORMED), (bearer(WRONG_KEY), INVALID)],
    ids=["no header", "wrong key"],
)
def test_auth_is_checked_before_json_decoding(client, fake_llm, headers, detail):
    response = client.post(
        "/chat",
        content=b'{"message": "Say hello"',  # truncated: not valid JSON
        headers={**headers, "Content-Type": "application/json"},
    )

    assert_rejected(response, detail)
    assert fake_llm.calls == []


@pytest.mark.parametrize(
    "headers",
    [{}, bearer(WRONG_KEY), {"Authorization": "garbage"}],
    ids=["no header", "wrong key", "malformed header"],
)
def test_health_stays_public(client, set_gateway_keys, headers):
    # Even with no keys configured at all, /health must answer.
    set_gateway_keys(None)

    response = client.get("/health", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "www-authenticate" not in response.headers


# --- Logging ---------------------------------------------------------------------


def test_keys_never_appear_in_logs(client, set_gateway_keys, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    too_short = "short-secret-value"
    near_miss = PRIMARY_KEY[:-1] + "X"
    # The too-short entry makes config.py log a warning about it.
    set_gateway_keys(f"{PRIMARY_KEY},{too_short},{SECONDARY_KEY}")

    requests = [
        ({"json": CHAT_BODY, "headers": bearer(PRIMARY_KEY)}, 200),
        ({"json": CHAT_BODY, "headers": bearer(SECONDARY_KEY)}, 200),
        ({"json": CHAT_BODY, "headers": bearer(WRONG_KEY)}, 401),
        ({"json": CHAT_BODY, "headers": bearer(near_miss)}, 401),
        ({"json": CHAT_BODY, "headers": {"Authorization": f"Basic {PRIMARY_KEY}"}}, 401),
        ({"json": CHAT_BODY, "headers": {"Authorization": f"Bearer {PRIMARY_KEY} x"}}, 401),
        ({"json": {"message": ""}, "headers": bearer(WRONG_KEY)}, 401),
    ]
    for kwargs, expected_status in requests:
        assert client.post("/chat", **kwargs).status_code == expected_status

    captured = capsys.readouterr()
    output = caplog.text + captured.out + captured.err

    # Make sure the log paths really ran, so the check below isn't vacuous.
    assert "Rejected request to /chat" in output
    assert "GATEWAY_API_KEYS entry #2 is shorter" in output

    for secret in (PRIMARY_KEY, SECONDARY_KEY, too_short, WRONG_KEY, near_miss):
        assert secret not in output
        # A partial key is a leak too.
        assert secret[: len(secret) // 2] not in output
