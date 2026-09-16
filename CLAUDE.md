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

Settings come from environment variables, with a `.env` file at the repo root as a fallback (real env vars take precedence): `GOOGLE_API_KEY` (required for `/chat`), `GATEWAY_API_KEYS` (comma-separated client keys, each at least 32 characters, sent as `Authorization: Bearer <key>`; list two while rotating; if empty, `/chat` rejects every request), `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`. `.env` is gitignored; `.env.example` is the committed template.

`GATEWAY_API_KEYS=...` must be a single unbroken line in `.env`, not appended onto the end of another variable's line; a line-glue mistake here previously caused every request to be silently rejected.

## Architecture

Request flow: `app/main.py` (app + `/health`) → `app/routers/chat.py` (`POST /chat`, Pydantic request/response models) → `app/services/llm_client.py` (the Gemini call).

Conventions to keep consistent when extending:

- **Settings are loaded once.** `app/config.py` calls `load_dotenv` at import time, and `get_settings()` is `lru_cache`d. Read config through `get_settings()`, not `os.getenv`. Changes to `.env` take effect only after a server restart.
- **The Gemini client is a lazy, cached singleton** (`_get_client()` in `llm_client.py`). The server still starts without an API key: `/health` reports `api_key_configured: false`, and `/chat` fails when it's called.
- **Error contract:** the service layer raises only `LLMError` (it wraps `genai_errors.APIError`, a missing key, and empty replies). Routers turn `LLMError` into HTTP `502` with the message in `detail`. Pydantic validation failures (e.g. an empty `message`) return `422`.
- **Async all the way down:** use `client.aio.models.*` so the event loop isn't blocked.
- Automatic function calling is explicitly disabled in `GenerateContentConfig` because no tools are registered. If you add tools, revisit that setting.

## Security posture

Auth: yes, rate limiting: not yet. `/chat` requires a key from `GATEWAY_API_KEYS` (see `app/auth.py`); `/health` stays public. There is still no content guardrail, and the service is meant to run on localhost only.

## Git Commit Policy

Never add a Co-Authored-By trailer, a "Generated with Claude Code" line, or any Claude/Anthropic attribution to git commit messages or PR descriptions.
