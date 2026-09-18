# AI Gateway

A minimal FastAPI service that forwards chat messages to the Google Gemini API.

## Structure

```
app/
  main.py              FastAPI entrypoint (app instance, router wiring, /health)
  config.py            Loads .env once and exposes typed settings
  auth.py              API key check for /chat
  rate_limit.py        Per-client sliding-window limit for /chat
  guardrails.py        PII and prompt-injection checks on /chat input,
                       PII and secret checks on the reply
  routing.py           Picks a model by message size, retries once on failure
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
   `RATE_LIMIT_REQUESTS` (default 5), `RATE_LIMIT_WINDOW_SECONDS` (default 60),
   `GUARDRAIL_CHECKS` (default `email,card,phone,injection`),
   `GEMINI_CHEAP_MODEL` (default `gemini-flash-lite-latest`),
   `ROUTING_TOKEN_THRESHOLD` (default 1000).
   Get a key from https://aistudio.google.com/apikey.

## Run

```bash
uvicorn app.main:app --reload
```

The server listens on http://127.0.0.1:8000. Interactive docs: http://127.0.0.1:8000/docs

The docs page has an **Authorize** button: paste a key from `GATEWAY_API_KEYS`
and `/chat` can be tried from the browser instead of curl. It only fills in the
`Authorization` header for you; the key is still checked by `app/auth.py`.

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model": "gemini-flash-latest",
  "cheap_model": "gemini-flash-lite-latest",
  "api_key_configured": true
}
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
`502` with a fixed `detail` of `Upstream model request failed.` — the real reason
is written to the server log, because the upstream error text can quote your
request back and name internal identifiers. A missing, empty, or over-long
`message` field (the cap is 16,000 characters) is a `422` naming the field and
the problem, but never repeating the value you sent. A missing or unknown API key
is a `401`.

Short messages are answered by a cheaper model and longer ones escalate: below
`ROUTING_TOKEN_THRESHOLD` estimated tokens (roughly characters / 4) the request
goes to `GEMINI_CHEAP_MODEL`, at or above it to `GEMINI_MODEL`. The `model` field
in the response always names the model that actually answered. If that call
fails with a retryable error, the request is retried **once** on the other tier
and no more; that retry costs you nothing extra against the rate limit.

Messages are screened before they reach Gemini. Anything that looks like an
email address, a payment card number or a phone number, or that resembles a
prompt-injection attempt, is rejected with a `400` naming the categories that
tripped — never quoting your message back. Turn individual checks off with
`GUARDRAIL_CHECKS`. The screening is deliberately shallow: see the "Notes"
section.

Replies are screened on the way back too. If the model's answer contains
something that looks like an email address, a card number or a phone number, or
contains one of your `GATEWAY_API_KEYS` values or the `GOOGLE_API_KEY`, the reply
is withheld: `/chat` answers `502` with a fixed `detail` of `Reply withheld by
the gateway.` and returns none of the text. The response is identical whichever
check tripped, so it names neither the matched content nor the category; the
server log records the category and the client, never the reply. The same
`GUARDRAIL_CHECKS` switches apply (the injection check is input-only), while the
secret check is always on. There is no redaction or partial answer, and a
withheld reply still costs you the upstream call and a rate-limit slot.

Each key gets 5 requests per 60 seconds by default. Over that, `/chat` answers
`429` with a `Retry-After` header saying how many seconds to wait. The limit is
per key, so one client running hot never affects another, and `/health` is never
limited.

Every `/chat` request, accepted or rejected, writes one JSON line to stdout:

```json
{"ts": "2026-09-18T10:15:02.114+00:00", "key_id": "3f9a1c0d2b7e4a61", "method": "POST", "path": "/chat", "status": 200, "latency_ms": 812.4, "model": "gemini-flash-lite-latest"}
```

`key_id` is a non-secret label for the client key, or `"unauthenticated"` when no
key was accepted. `model` is `null` when no model answered. The line never
contains the key, your message, the reply, or the query string. It goes to stdout
only; shipping it anywhere else is up to whatever runs the process.

## Notes

`/chat` requires an API key, is rate limited per key, screens input for PII and
prompt injection, and screens replies for PII and secrets. Rate-limit state is
held in memory in a single process — running more than one worker multiplies the
effective limit.

The fallback is between two Gemini tiers on one API key. It covers a single
model being slow or briefly unavailable; it does not cover Google being down,
the key being revoked, or the quota being spent, because both tiers share the
same key and account. A real multi-provider fallback would need a second vendor
and a second paid key, which this project deliberately does not require.

The input screening is a first pass, not a guarantee. PII detection is regular
expressions plus a Luhn and issuer-prefix check; it does not cover SSNs, IBANs,
passports, addresses or names. The prompt-injection check matches a handful of
known English phrasings and is defeated by translation, base64, spacing tricks
or simply rewording. Treat a message that passes as "nothing obvious found",
never as "safe".

The reply screening is a tripwire rather than a boundary, and it is worth
knowing why. Nothing secret is ever sent to the model — there is no system
prompt, and neither key is part of the request — so a key can only come back if
you pasted it into your message, by which point it has already reached Google.
PII in a reply is example or invented data, and ordinary examples are withheld
along with the rest: an answer containing `user@example.com`, the
`4242 4242 4242 4242` test card or `555-123-4567` trips it. It also reuses the
same shallow detectors, so a reply asked for in another shape (spaced out,
spelled out, base64) walks past it. It catches accidents, and it earns more the
day a system prompt, tools or retrieval give the model something worth leaking.
Do not expose this beyond localhost as-is.
