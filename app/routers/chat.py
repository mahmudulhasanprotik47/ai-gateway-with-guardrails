"""POST /chat — forward a message to the model and return its reply."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.config import get_settings
from app.services.llm_client import LLMError, generate_reply

# Every route on this router requires an API key; /health lives on the app
# itself and stays public.
router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's prompt.")


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
