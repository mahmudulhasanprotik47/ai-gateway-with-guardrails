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

Settings come from environment variables, with a `.env` file at the repo root as a fallback (real env vars take precedence): `GOOGLE_API_KEY` (required for `/chat`), `GATEWAY_API_KEYS` (comma-separated client keys, each at least 32 characters, sent as `Authorization: Bearer <key>`; list two while rotating; if empty, `/chat` rejects every request), `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`, `RATE_LIMIT_REQUESTS` (default 5; `0` closes `/chat` to everyone) and `RATE_LIMIT_WINDOW_SECONDS` (default 60), which bound `/chat` per client key. `.env` is gitignored; `.env.example` is the committed template.

`GATEWAY_API_KEYS=...` must be a single unbroken line in `.env`, not appended onto the end of another variable's line; a line-glue mistake here previously caused every request to be silently rejected.

## Architecture

Request flow: `app/main.py` (app + `/health`) → `app/routers/chat.py` (`POST /chat`, Pydantic request/response models; router dependencies run `app/auth.py` then `app/rate_limit.py`) → `app/services/llm_client.py` (the Gemini call).

Conventions to keep consistent when extending:

- **Settings are loaded once.** `app/config.py` calls `load_dotenv` at import time, and `get_settings()` is `lru_cache`d. Read config through `get_settings()`, not `os.getenv`. Changes to `.env` take effect only after a server restart.
- **The Gemini client is a lazy, cached singleton** (`_get_client()` in `llm_client.py`). The server still starts without an API key: `/health` reports `api_key_configured: false`, and `/chat` fails when it's called.
- **Error contract:** the service layer raises only `LLMError` (it wraps `genai_errors.APIError`, a missing key, and empty replies). Routers turn `LLMError` into HTTP `502` with the message in `detail`. Pydantic validation failures (e.g. an empty `message`) return `422`.
- **Async all the way down:** use `client.aio.models.*` so the event loop isn't blocked.
- **Rate limiting is per authenticated client, in memory.** `app/rate_limit.py` keeps a sliding window log (a deque of `time.monotonic` timestamps) per `key_id`. `enforce_rate_limit` takes `require_api_key` as a *sub-dependency*, which is what guarantees 401 beats 429 and that the limiter always has an identity; it must stay `async def`, or FastAPI runs it in a threadpool and the check-then-append stops being atomic. A rejected request is never appended to the bucket. Rejections are HTTP `429` with a `Retry-After` header.
- Automatic function calling is explicitly disabled in `GenerateContentConfig` because no tools are registered. If you add tools, revisit that setting.

## Security posture

Auth: yes. Rate limiting: yes, per client key, in memory. `/chat` requires a key from `GATEWAY_API_KEYS` (see `app/auth.py`) and is limited per key (see `app/rate_limit.py`); `/health` stays public and unlimited. `ChatRequest.message` is capped at `MAX_MESSAGE_LENGTH` (16,000 chars), because the limiter counts requests while Gemini bills tokens. There is still no content guardrail, and the service is meant to run on localhost only.

Three known gaps, documented on purpose rather than fixed:

- **Single process only.** Rate-limit state is a module-level dict. Under `uvicorn --workers N` or several instances, each process counts separately and the effective limit becomes N x the configured one. Correct multi-process limiting needs a shared store (Redis `INCR`+`EXPIRE`, or a sorted set).
- **Per-client, not global.** N clients x the per-client limit all land on one `GOOGLE_API_KEY`, so this protects clients from each other, not the shared upstream quota from all of them. Two keys listed during a rotation give one operator 2x the limit.
- **The body is read before the rejection.** Starlette buffers and JSON-parses the request body before dependencies run, so a 401'd or 429'd caller still costs a full read and parse. This limits work sent to Gemini, not work done by the process; a `Content-Length` guard is the follow-up.

## Git Commit Policy

Never add a Co-Authored-By trailer, a "Generated with Claude Code" line, or any Claude/Anthropic attribution to git commit messages or PR descriptions.
