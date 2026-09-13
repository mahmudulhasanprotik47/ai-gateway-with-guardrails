# AI Gateway

A minimal FastAPI service that forwards chat messages to the Google Gemini API.

## Structure

```
app/
  main.py              FastAPI entrypoint (app instance, router wiring, /health)
  config.py            Loads .env once and exposes typed settings
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

   Optional overrides: `GEMINI_MODEL` (default `gemini-flash-latest`), `APP_NAME`.
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
  -H "Content-Type: application/json" \
  -d '{"message": "Say hello in one sentence."}'
```

```json
{ "reply": "Hello there!", "model": "gemini-flash-latest" }
```

Errors from the upstream model (missing key, quota, bad model name) come back as
`502` with the reason in `detail`. A missing or empty `message` field is a `422`.

## Notes

There is no authentication, rate limiting, or content guardrail in front of this
service yet — do not expose it beyond localhost as-is.
