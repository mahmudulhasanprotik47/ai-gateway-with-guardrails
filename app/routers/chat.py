"""POST /chat — forward a message to the model and return its reply."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import AuthenticatedClient, docs_bearer_scheme, require_api_key
from app.config import MAX_MESSAGE_LENGTH, Settings, get_settings
from app.guardrails import check_message
from app.rate_limit import enforce_rate_limit
from app.routing import generate_with_fallback
from app.services.llm_client import ContentBlocked, LLMError

# Every route on this router requires an API key and is rate limited per key;
# /health lives on the app itself and stays public. require_api_key is listed
# here for the docs, and is also a sub-dependency of enforce_rate_limit, which
# is what guarantees 401 beats 429. FastAPI caches it, so it runs once.
router = APIRouter(
    tags=["chat"],
    dependencies=[
        Depends(require_api_key),
        Depends(enforce_rate_limit),
        # Listed last and purely for the docs UI: it never rejects anything
        # (see docs_bearer_scheme), it just puts the scheme in the OpenAPI
        # document so /docs grows an Authorize button.
        Depends(docs_bearer_scheme),
    ],
)

# Requests are counted, not tokens, so a cap on the prompt is what bounds the
# cost of a single call. ~4k tokens: far more than a chat turn needs. Defined in
# config.py (the routing threshold is sanity-checked against it) and re-exported
# here, where callers and tests have always looked for it.
__all__ = ["MAX_MESSAGE_LENGTH", "ChatRequest", "ChatResponse", "chat", "router"]

# What a caller is told when the upstream call fails, whichever LLMError it was.
# Every one of those messages is written for operators: one names an
# environment variable, another describes the model's behaviour. None of them
# is the caller's business, and the API error text can quote the request back.
_UPSTREAM_FAILURE_DETAIL = "Upstream model request failed."


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's prompt.",
    )


class ChatResponse(BaseModel):
    reply: str
    model: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    # Both are already resolved for the router dependencies above; FastAPI
    # caches them per request, so asking again costs nothing and gives the
    # handler the identity it logs and the settings the guardrails read.
    # Taking settings via Depends rather than calling get_settings() here is
    # what lets tests override them: get_settings is lru_cached, so a direct
    # call ignores app.dependency_overrides entirely.
    client: AuthenticatedClient = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    # Runs after auth (401) and the rate limiter (429), and after Pydantic has
    # validated the body (422). A rejection here consumes a rate-limit slot,
    # exactly as a 422 or a 502 already does.
    check_message(payload.message, settings, client.key_id)

    try:
        reply, model_used = await generate_with_fallback(
            payload.message, settings, client.key_id
        )
    except ContentBlocked as exc:
        # The model refused the caller's input, so this is the caller's problem.
        # This message is ours and safe to show; it quotes nothing.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMError as exc:
        # Deliberately not str(exc): see _UPSTREAM_FAILURE_DETAIL. llm_client
        # has already logged what actually went wrong.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_UPSTREAM_FAILURE_DETAIL,
        ) from exc

    # Whichever model answered, including after a fallback - not the configured
    # default, which is only one of the two tiers.
    return ChatResponse(reply=reply, model=model_used)
