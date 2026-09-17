# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A minimal FastAPI service that forwards chat messages to Google Gemini via the `google-genai` SDK. The local `.venv` uses Python 3.14.

## Commands

Run `.venv\Scripts\Activate.ps1` in every new terminal; until you do, `uvicorn` and `pytest` won't work.

```bash
# Setup (Windows PowerShell shown; use `source .venv/bin/activate` on macOS/Linux)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test deps (pytest, httpx2); includes requirements.txt

# Run the dev server (http://127.0.0.1:8000, docs at /docs)
uvicorn app.main:app --reload

# Run the tests (settings are overridden and generate_reply is faked: no real Gemini calls)
pytest

# Smoke test
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Authorization: Bearer <one of GATEWAY_API_KEYS>" -H "Content-Type: application/json" -d '{"message": "Say hello"}'
```

Tests live in `tests/` (pytest). There is no linter or formatter configured yet.

## Configuration

Settings come from environment variables, with a `.env` file at the repo root as a fallback (real env vars take precedence): `GOOGLE_API_KEY` (required for `/chat`), `GATEWAY_API_KEYS` (comma-separated client keys, each at least 32 characters, sent as `Authorization: Bearer <key>`; list two while rotating; if empty, `/chat` rejects every request), `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`, `RATE_LIMIT_REQUESTS` (default 5; `0` closes `/chat` to everyone) and `RATE_LIMIT_WINDOW_SECONDS` (default 60), which bound `/chat` per client key. `GUARDRAIL_CHECKS` (comma-separated; unset means all of `email,card,phone,injection`, set-but-empty means none). `GEMINI_CHEAP_MODEL` (default `gemini-flash-lite-latest`) and `ROUTING_TOKEN_THRESHOLD` (default 1000) drive complexity routing; setting the two model names equal disables the fallback retry. `.env` is gitignored; `.env.example` is the committed template.

`GATEWAY_API_KEYS=...` must be a single unbroken line in `.env`, not appended onto the end of another variable's line; a line-glue mistake here previously caused every request to be silently rejected.

## Architecture

Request flow: `app/main.py` (app + `/health`) → `app/routers/chat.py` (`POST /chat`, Pydantic request/response models; router dependencies run `app/auth.py` then `app/rate_limit.py`, then the handler body calls `app/guardrails.py`, then `app/routing.py`) → `app/services/llm_client.py` (the Gemini call).

Conventions to keep consistent when extending:

- **Settings are loaded once.** `app/config.py` calls `load_dotenv` at import time, and `get_settings()` is `lru_cache`d. Read config through `get_settings()`, not `os.getenv`. Changes to `.env` take effect only after a server restart.
- **The Gemini client is a lazy, cached singleton** (`_get_client()` in `llm_client.py`). The server still starts without an API key: `/health` reports `api_key_configured: false`, and `/chat` fails when it's called.
- **Error contract:** the service layer raises only `LLMError` (it wraps `genai_errors.APIError`, a missing key, and empty replies). Routers turn `LLMError` into HTTP `502`. Pydantic validation failures (e.g. an empty or over-long `message`) return `422`.
- **Rejections never quote the request back.** Two places leaked it. `genai_errors.APIError` stringifies as `"<code> <status>. <response_json>"`, and that body can echo request content plus model and project identifiers, so the `502` detail is the fixed string `Upstream model request failed.` and the real error is logged instead. Pydantic records the rejected value in `input`, and FastAPI serializes the error list as-is, so `validation_exception_handler` in `main.py` strips `input` and `ctx` from every `422`; `type`, `loc` and `msg` still say what was wrong. `tests/test_error_contract.py` pins both.
- **Async all the way down:** use `client.aio.models.*` so the event loop isn't blocked.
- **Rate limiting is per authenticated client, in memory.** `app/rate_limit.py` keeps a sliding window log (a deque of `time.monotonic` timestamps) per `key_id`. `enforce_rate_limit` takes `require_api_key` as a *sub-dependency*, which is what guarantees 401 beats 429 and that the limiter always has an identity; it must stay `async def`, or FastAPI runs it in a threadpool and the check-then-append stops being atomic. A rejected request is never appended to the bucket. Rejections are HTTP `429` with a `Retry-After` header.
- **Input guardrails are a plain call, not a dependency.** `check_message` in `app/guardrails.py` runs on the first line of the `chat()` handler, so it lands after auth (401), the rate limiter (429) and Pydantic (422), and a rejection consumes a rate-limit slot like any other. It raises `HTTPException(400)` directly, the way `auth.py` and `rate_limit.py` do. The cost of not being a router dependency: it is not inherited, so a future route must call it itself.
- **Card detection needs the issuer prefix, not just Luhn.** Testing every 13-19 digit window with Luhn alone flags ~65% of random 16-digit numbers, because some window almost always passes; order and invoice numbers would be rejected more often than not. Windows must also start with a real issuer prefix and be a length that issuer uses, which measures at ~7.5%. `tests/test_guardrails.py::test_card_windows_do_not_swamp_ordinary_numbers` pins it.
- **Normalize before matching.** `guardrails.normalize` applies NFKC and strips zero-width characters. Without it a fullwidth or zero-width-split card number, and an injection phrase with an invisible character inside a keyword, both walk straight through.
- Automatic function calling is explicitly disabled in `GenerateContentConfig` because no tools are registered. If you add tools, revisit that setting — and re-read the injection guardrail's docstring, which currently says the check defends nothing because there is no system prompt or tool to hijack.
- **Routing picks the tier and owns the retry.** `app/routing.py` estimates tokens as `len(message) // 4` - an estimate, wrong for CJK and code, used only to choose a model - and sends anything below `ROUTING_TOKEN_THRESHOLD` to the cheap tier, escalating at or above it. A failed call is retried **exactly once** on the other tier, structurally: the retry sits after the `except` block, so there is no loop or counter to tune into one. Retry policy lives here rather than in `llm_client.py` so that module stays the thin transport wrapper its docstring claims to be.
- **Not every failure is worth retrying.** `LLMError` carries a required `retryable` keyword - required, not defaulted, so a new raise site cannot become retryable by omission. Only 5xx and empty replies are retried. `429` is the shared account's quota and retrying compounds an exhaustion already in progress; `403` and `400` fail identically on either tier; `404` is a mistyped model id, and retrying it would hide the typo behind a fallback that always works. `ContentBlocked` is never retried at all - identical content against identical safety settings will not fare better elsewhere.
- **`generate_reply` takes the model explicitly.** It used to read `settings.gemini_model` itself, which made the response's `model` field a lie the moment two tiers existed. The caller that chose the model is now the one that reports it. Consequence for tests: `generate_reply` is imported by name into `routing.py`, so `tests/conftest.py` patches **`routing.generate_reply`** - patching `app.routers.chat` intercepts nothing, because the router no longer imports it.
- **Read settings through `Depends(get_settings)` in route handlers.** `get_settings` is `lru_cache`d, so a direct call inside a handler ignores `app.dependency_overrides` and silently serves whatever was cached first. Both `chat()` and `health()` had this bug; both now take settings as a dependency.
- **Safety settings are explicit** in `llm_client.py`: `BLOCK_MEDIUM_AND_ABOVE` on the four standard harm categories, `CIVIC_INTEGRITY` deliberately unset. A module constant, not a setting. A blocked *prompt* (`prompt_feedback.block_reason`) raises `ContentBlocked` → `400`, because the caller's input was refused; a withheld *reply* (`finish_reason` in the refusal set) stays `LLMError` → `502`. `ContentBlocked` is a sibling of `LLMError`, not a subclass, so the router's catch order cannot silently regress the mapping.

## Security posture

Auth: yes. Rate limiting: yes, per client key, in memory. `/chat` requires a key from `GATEWAY_API_KEYS` (see `app/auth.py`) and is limited per key (see `app/rate_limit.py`); `/health` stays public and unlimited. `ChatRequest.message` is capped at `MAX_MESSAGE_LENGTH` (16,000 chars), because the limiter counts requests while Gemini bills tokens. Input guardrails reject obvious PII and casual prompt-injection attempts with a `400` (see `app/guardrails.py`). The service is meant to run on localhost only.

Five known gaps, documented on purpose rather than fixed:

- **Single process only.** Rate-limit state is a module-level dict. Under `uvicorn --workers N` or several instances, each process counts separately and the effective limit becomes N x the configured one. Correct multi-process limiting needs a shared store (Redis `INCR`+`EXPIRE`, or a sorted set).
- **Per-client, not global.** N clients x the per-client limit all land on one `GOOGLE_API_KEY`, so this protects clients from each other, not the shared upstream quota from all of them. Two keys listed during a rotation give one operator 2x the limit.
- **The guardrails are shallow, and say so.** PII detection is regex plus Luhn plus an issuer-prefix check: no SSNs, IBANs, passports, addresses or names, and no context. The injection check matches a few English phrasings and is beaten by translation, base64, spacing or rephrasing; `tests/test_guardrails.py::test_documented_bypasses_are_not_caught` pins exactly how thin it is. It is a first pass, not a security boundary, and today it is mostly forward-looking — there is no system prompt or tool for an injection to hijack yet.
- **The fallback is one vendor, one key.** `app/routing.py` retries on the other Gemini *tier*, which rides out one model being slow, overloaded or briefly unavailable. It does not survive Google being down, the key being revoked, or the account's quota being spent, because both tiers sit behind the same `GOOGLE_API_KEY` and the same account. Real multi-provider fallback needs a second vendor and a second paid key, which this project has deliberately never required. The fallback also means one admitted request can cost two upstream calls; `DEFAULT_RATE_LIMIT_REQUESTS` is no longer a 1:1 ratio with upstream calls, and its comment in `config.py` spells out the 2x worst case.
- **The body is read before the rejection.** Starlette buffers and JSON-parses the request body before dependencies run, so a 401'd or 429'd caller still costs a full read and parse. This limits work sent to Gemini, not work done by the process; a `Content-Length` guard is the follow-up.

## Git Commit Policy

Never add a Co-Authored-By trailer, a "Generated with Claude Code" line, or any Claude/Anthropic attribution to git commit messages or PR descriptions.
