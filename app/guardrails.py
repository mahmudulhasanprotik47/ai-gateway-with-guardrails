"""Guardrails for /chat: input PII and injection checks, output PII and secrets.

On the way in, two checks, both deliberately shallow, both rejecting rather
than redacting so the caller learns why their message did not go through. On
the way out, `check_reply` runs the same PII detectors on the model's reply plus
a secret check, and withholds the reply without saying why; see its docstring
for how little that defends today.

**PII is pattern matching, not DLP.** It looks for email addresses, card-shaped
numbers that pass a Luhn check, and punctuated phone numbers. It does not look
for SSNs, IBANs, passport numbers, postal addresses, dates of birth, or names,
and it cannot reason about context. Treat a pass as "no obvious card number in
here", never as "this message is clean".

**The injection check is a heuristic first pass and not a security boundary.**
It matches a handful of well-known English phrasings. Translation, base64,
homoglyphs, typos, added spacing, or simply rewriting the request all walk
straight past it. It raises the cost of a casual attempt and nothing more.

It is also, today, mostly forward-looking: `generate_reply` sends the caller's
message as the entire prompt, with no system instruction to override, no tools
to hijack, and automatic function calling disabled. There is nothing for an
injected instruction to take over yet. The check earns its place the moment a
system prompt or a tool is added, which is why it ships now rather than later.

Matched text is never echoed - not into the response, not into the log. The
whole premise is that the message may hold something sensitive, so quoting it
back into an error body or a log file would be the leak this is meant to
prevent. Callers get category names; operators get category names plus the
key_id.
"""

import logging
import re
import unicodedata

from fastapi import HTTPException, status

from app.config import Settings

logger = logging.getLogger(__name__)

# Zero-width and bidi characters, stripped before matching. Without this,
# "4111<ZWSP>1111..." is not a digit run and sails through every pattern below.
_INVISIBLE_RE = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")

_NON_DIGITS_RE = re.compile(r"\D")

# A run of digits that may contain spaces or hyphens, long enough to hold a
# 13-digit card. Bounded and un-nested, so matching stays linear.
_CARD_CANDIDATE_RE = re.compile(r"[0-9][0-9 \-]{11,}[0-9]")

# Issuer prefix -> the lengths that issuer actually uses. Luhn alone is not
# enough once every window of a digit run is tested: ~65% of random 16-digit
# numbers contain SOME 13-19 digit window that passes Luhn, so an order or
# invoice number would be rejected more often than not. Requiring the window to
# also start like a real card and be a length that issuer really uses takes
# that to ~7.5% while still catching every test PAN, including one with extra
# digits stuck on the front. Measured, not guessed; see test_card_windows.
#
# No "^" anchors: these are applied with pattern.match(digits, start), which
# already anchors at `start`, whereas "^" only ever matches at index 0 and would
# silently disable every window after the first.
_CARD_FORMATS = (
    (re.compile(r"4"), (13, 16, 19)),  # Visa
    (re.compile(r"(?:5[1-5]|2[2-7])"), (16,)),  # Mastercard
    (re.compile(r"3[47]"), (15,)),  # American Express
    (re.compile(r"(?:6011|65|64[4-9])"), (16,)),  # Discover
    (re.compile(r"3(?:0[0-5]|[689])"), (14,)),  # Diners Club
    (re.compile(r"35"), (16,)),  # JCB
)

# Crude on purpose, and never the sole gate: the literal "@" pre-check in
# _has_email is what stops this running on text that cannot contain an address.
# Both sides are bounded (RFC 5321's 64-octet local part and 253-octet domain),
# which is what keeps a search linear: unbounded "+" made it quadratic on a
# crafted run of one character then an "@" (64k took ~24s). That used to be
# capped by MAX_MESSAGE_LENGTH, but replies have no such cap.
_EMAIL_RE = re.compile(r"[^@\s]{1,64}@[^@\s]{1,253}\.[A-Za-z]{2,}")

# Punctuation is required, so bare digit runs - years, quantities, order
# numbers - are not phone numbers. This is the least precise of the three.
_PHONE_RE = re.compile(
    r"\+\d[\d\-.\s()]{7,}\d"  # +44 20 7946 0958
    r"|\(\d{3}\)[\s.\-]*\d{3}[\s.\-]*\d{4}"  # (555) 010-1234
    r"|\b\d{3}[.\-]\d{3}[.\-]\d{4}\b"  # 555-010-1234
)

# Four families. Every quantifier is bounded and none is nested.
_INJECTION_RES = (
    # Instruction override. Deliberately does not require an "instructions"
    # noun, which is what makes "ignore the previous paragraph" a known false
    # positive: the heuristic cannot tell that apart from an attack without
    # understanding the sentence. See tests/test_guardrails.py.
    re.compile(
        r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}"
        r"\b(?:previous|prior|above|earlier|preceding)\b",
        re.IGNORECASE,
    ),
    # Role impersonation: a line that opens like a chat transcript turn.
    re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*:", re.IGNORECASE),
    # Role impersonation: raw chat-template markers.
    re.compile(r"<\|im_start\|>|\[INST\]|<<SYS>>", re.IGNORECASE),
    # Prompt extraction.
    re.compile(
        r"\b(?:reveal|show|print|repeat|output|display)\b[^.\n]{0,30}"
        r"\byour\b[^.\n]{0,30}\b(?:system prompt|instructions|rules|prompt)\b",
        re.IGNORECASE,
    ),
    # Named jailbreak handles.
    re.compile(r"\b(?:developer mode|dan mode|do anything now)\b", re.IGNORECASE),
)

# What the caller is told, per category. No matched text, ever.
_PII_LABELS = {
    "email": "an email address",
    "card": "a card number",
    "phone": "a phone number",
}


def normalize(message: str) -> str:
    """Fold compatibility forms and drop invisible characters before matching.

    NFKC turns fullwidth digits into ASCII ones, so a fullwidth card number is
    still a card number. Stripping zero-width characters stops the oldest trick
    in the book, splitting a number with an invisible space.
    """
    return _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", message))


def _luhn_ok(digits: str) -> bool:
    """Standard Luhn checksum. This is what keeps order numbers out of the net."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _has_email(text: str) -> bool:
    if "@" not in text:  # the pre-check: no address is possible without one
        return False
    return _EMAIL_RE.search(text) is not None


def _has_card(text: str) -> bool:
    """True if a digit run contains a window that looks like a real card number.

    A window qualifies only if it starts with a known issuer prefix, is a length
    that issuer uses, and passes Luhn. Every start position is tried rather than
    just the whole run, because a card rarely sits alone: "ref 14111111111111111"
    leaves a 17-digit run once separators come out, and testing only the full run
    would miss the 16-digit card inside it.

    Windows are never copied as `digits[start:]`: that is a fresh copy of the
    rest of the run per start position, quadratic on a long run. Matching at
    `start` and slicing only `length` digits keeps the scan linear.
    """
    if not any(char.isdigit() for char in text):  # the pre-check
        return False
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = _NON_DIGITS_RE.sub("", match.group())
        for start in range(len(digits)):
            for prefix, lengths in _CARD_FORMATS:
                if not prefix.match(digits, start):
                    continue
                for length in lengths:
                    end = start + length
                    if end <= len(digits) and _luhn_ok(digits[start:end]):
                        return True
    return False


def _has_phone(text: str) -> bool:
    if not any(char.isdigit() for char in text):  # the pre-check
        return False
    return _PHONE_RE.search(text) is not None


def _has_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_RES)


_DETECTORS = {
    "email": _has_email,
    "card": _has_card,
    "phone": _has_phone,
    "injection": _has_injection,
}


def find_violations(message: str, checks: frozenset[str]) -> tuple[str, ...]:
    """Return the names of every enabled check the message trips, in order."""
    text = normalize(message)
    return tuple(
        name
        for name in ("email", "card", "phone", "injection")
        if name in checks and _DETECTORS[name](text)
    )


def _build_detail(categories: tuple[str, ...]) -> str:
    """Say which categories tripped, quoting none of the message."""
    clauses = []
    pii = [_PII_LABELS[name] for name in categories if name in _PII_LABELS]
    if pii:
        clauses.append(f"it appears to contain {', '.join(pii)}")
    if "injection" in categories:
        clauses.append("it resembles a prompt-injection attempt")

    detail = f"Message rejected: {'; '.join(clauses)}."
    if pii:
        detail += " Remove personal data and try again."
    return detail


def check_message(message: str, settings: Settings, key_id: str) -> None:
    """Reject a message that trips an enabled guardrail, with HTTP 400.

    Raises `HTTPException` directly rather than a custom error, the way
    `app/auth.py` and `app/rate_limit.py` already do, so the router needs no
    translation clause.

    Called from the handler body rather than as a router dependency: a
    dependency that inspects the body would have to redeclare `ChatRequest`,
    and would sit alongside Pydantic's own validation rather than cleanly after
    it. Being in the handler means the 422 still comes first and this runs only
    on a request that already parsed. It also means any future route has to
    call this for itself - it is not inherited the way the router-level auth
    and rate-limit dependencies are.
    """
    categories = find_violations(message, settings.guardrail_checks)
    if not categories:
        return

    # Category names and the key_id only; the message itself is never logged.
    logger.warning(
        "Guardrail rejected a request from client %s: %s",
        key_id,
        ", ".join(categories),
    )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=_build_detail(categories),
    )


# --- Output ---------------------------------------------------------------------

# Internal only: names this path in the log line so it cannot be mistaken for an
# input rejection. Never part of the response.
_REASON = "output_guardrail"

# What the caller is told, whichever check tripped. A constant, not a template:
# both earlier leaks in this project were rejection bodies built from the
# rejected content, and _build_detail is deliberately not reused because it
# interpolates category names.
_REPLY_WITHHELD_DETAIL = "Reply withheld by the gateway."


def _leaks_secret(text: str, settings: Settings) -> bool:
    """True if `text` contains a configured gateway key or the Google API key.

    Whole values only. A prefix or fragment match would let a caller ask the
    model to echo candidate prefixes and walk a key out one character at a time;
    a whole-value hit only confirms a guess the caller already made in full.
    Empty values are skipped because "" is in every string.
    """
    secrets = (*settings.api_keys, settings.google_api_key)
    return any(secret and secret in text for secret in secrets)


def check_reply(reply: str, settings: Settings, key_id: str) -> None:
    """Withhold a model reply that carries PII or a secret, with HTTP 502.

    A hard block: no redaction, no pass-through. PII reuses the input detectors
    and follows `GUARDRAIL_CHECKS`, minus the injection heuristic, which judges
    what a caller asks and not what a model answers. The secret check has no
    switch; an exact match on a 32+ character value has no false-positive story.

    502 rather than 400 for the reason a reply Gemini itself withholds is a 502:
    the caller's input already passed `check_message`, so there is nothing to
    blame them for. The detail is its own constant, distinct from the router's
    upstream-failure string, so a block is not mistaken for an outage.

    **A tripwire, not a boundary.** Nothing secret is ever sent to the model:
    there is no system prompt, and neither key appears in the request. So today
    a secret can only reach a reply if the caller pasted it into the prompt -
    at which point it has already gone to Google - and PII in a reply is
    fabricated or example data. A caller who wants content out asks for it
    spaced, spelled out, or base64'd, and this sees none of it. It exists for
    accidental leakage, and for the day a system prompt, tools, or retrieval
    give the model something worth leaking; a system prompt's text joins
    `secrets` in `_leaks_secret` then, with the same caveat about paraphrase.

    Called from the handler after routing returns, never inside it, so a block
    cannot trigger the tier fallback. Like `check_message`, a future route has
    to call this itself.
    """
    categories = find_violations(reply, settings.guardrail_checks - {"injection"})
    # Normalized for the same reason the PII detectors are: a fullwidth or
    # zero-width-split key is still the key. NFKC leaves key characters alone.
    if _leaks_secret(normalize(reply), settings):
        categories += ("secret",)
    if not categories:
        return

    # Reason code, category names and the caller's key_id. Never the reply, a
    # matched value, or which configured key matched.
    logger.warning(
        "Output guardrail (%s) withheld a reply to client %s: %s",
        _REASON,
        key_id,
        ", ".join(categories),
    )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=_REPLY_WITHHELD_DETAIL,
    )
