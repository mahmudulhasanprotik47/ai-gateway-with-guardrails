# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A minimal FastAPI service that forwards chat messages to Google Gemini via the `google-genai` SDK. The local `.venv` uses Python 3.14.

## Commands

```bash
# Setup (Windows PowerShell shown; use `source .venv/bin/activate` on macOS/Linux)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the dev server (http://127.0.0.1:8000, docs at /docs)
uvicorn app.main:app --reload

# Smoke test
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"message": "Say hello"}'
```

There is no test suite, linter, or formatter configured yet.

## Configuration

Settings come from environment variables, with a `.env` file at the repo root as a fallback (real env vars take precedence): `GOOGLE_API_KEY` (required for `/chat`), `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`. `.env` is gitignored; `.env.example` is the committed template.

## Architecture

Request flow: `app/main.py` (app + `/health`) → `app/routers/chat.py` (`POST /chat`, Pydantic request/response models) → `app/services/llm_client.py` (the Gemini call).

Conventions to keep consistent when extending:

- **Settings are loaded once.** `app/config.py` calls `load_dotenv` at import time, and `get_settings()` is `lru_cache`d. Read config through `get_settings()`, not `os.getenv`. Changes to `.env` take effect only after a server restart.
- **The Gemini client is a lazy, cached singleton** (`_get_client()` in `llm_client.py`). The server still starts without an API key: `/health` reports `api_key_configured: false`, and `/chat` fails when it's called.
- **Error contract:** the service layer raises only `LLMError` (it wraps `genai_errors.APIError`, a missing key, and empty replies). Routers turn `LLMError` into HTTP `502` with the message in `detail`. Pydantic validation failures (e.g. an empty `message`) return `422`.
- **Async all the way down:** use `client.aio.models.*` so the event loop isn't blocked.
- Automatic function calling is explicitly disabled in `GenerateContentConfig` because no tools are registered. If you add tools, revisit that setting.

## Security posture

There is no authentication, rate limiting, or content guardrail yet. The service is meant to run on localhost only.
