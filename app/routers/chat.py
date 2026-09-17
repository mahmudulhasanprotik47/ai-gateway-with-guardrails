"""POST /chat — forward a message to the model and return its reply."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.config import get_settings
from app.rate_limit import enforce_rate_limit
from app.services.llm_client import LLMError, generate_reply

# Every route on this router requires an API key and is rate limited per key;
# /health lives on the app itself and stays public. require_api_key is listed
# here for the docs, and is also a sub-dependency of enforce_rate_limit, which
# is what guarantees 401 beats 429. FastAPI caches it, so it runs once.
router = APIRouter(
    tags=["chat"],
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)

# Requests are counted, not tokens, so a cap on the prompt is what bounds the
# cost of a single call. ~4k tokens: far more than a chat turn needs.
MAX_MESSAGE_LENGTH = 16_000


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
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        reply = await generate_reply(payload.message)
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChatResponse(reply=reply, model=get_settings().gemini_model)
