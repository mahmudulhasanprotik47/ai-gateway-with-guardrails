# AI Gateway

A minimal FastAPI service that forwards chat messages to the Google Gemini API.

## Structure

```
app/
  main.py              FastAPI entrypoint (app instance, router wiring, /health)
  config.py            Loads .env once and exposes typed settings
  auth.py              API key check for /chat
  rate_limit.py        Per-client sliding-window limit for /chat
  routers/chat.py      POST /chat
  services/llm_client.py  Gemini call via the google-genai SDK
requirements.txt
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Put your key in `.env` at the repo root:

   ```
   GOOGLE_API_KEY=your-key-here
   ```

   You also need at least one client key in `GATEWAY_API_KEYS` (comma-separated,
   each at least 32 characters) or `/chat` will reject every request. See
   `.env.example`.

   Optional overrides: `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`,
   `RATE_LIMIT_REQUESTS` (default 5), `RATE_LIMIT_WINDOW_SECONDS` (default 60).
   Get a key from https://aistudio.google.com/apikey.

## Run

```bash
uvicorn app.main:app --reload
```

The server listens on http://127.0.0.1:8000. Interactive docs: http://127.0.0.1:8000/docs

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{ "status": "ok", "model": "gemini-flash-latest", "api_key_configured": true }
```

### `POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer <one of GATEWAY_API_KEYS>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Say hello in one sentence."}'
```

```json
{ "reply": "Hello there!", "model": "gemini-flash-latest" }
```

Errors from the upstream model (missing key, quota, bad model name) come back as
`502` with the reason in `detail`. A missing, empty, or over-long `message` field
(the cap is 16,000 characters) is a `422`. A missing or unknown API key is a `401`.

Each key gets 5 requests per 60 seconds by default. Over that, `/chat` answers
`429` with a `Retry-After` header saying how many seconds to wait. The limit is
per key, so one client running hot never affects another, and `/health` is never
limited.

## Notes

`/chat` requires an API key and is rate limited per key. There is still no content
guardrail, and rate-limit state is held in memory in a single process — running
more than one worker multiplies the effective limit. Do not expose this beyond
localhost as-is.
