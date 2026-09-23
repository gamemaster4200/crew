from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from openai_client import ask_model

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"

app = FastAPI(title="CREW", version="0.0.1")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


class ChatResponse(BaseModel):
    answer: str
    model: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.0.1"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        answer, model = await ask_model(message)
        return ChatResponse(answer=answer, model=model)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # v0.0.1 intentionally keeps error handling simple but visible.
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API request failed: {exc}",
        ) from exc