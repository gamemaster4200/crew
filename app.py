import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from panel_engine import SUPPORTED_MODELS, SUPPORTED_ROLES
from reviewer import run_comparison

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"

Role = Literal["Solver", "Critic", "Improver", "Integrator"]
Model = Literal["gpt-5.6-luna", "gpt-5.6-sol"]

app = FastAPI(title="CREW", version="0.0.3")


class AgentConfig(BaseModel):
    role: Role
    model: Model


class PanelConfig(BaseModel):
    agents: list[AgentConfig] = Field(min_length=1, max_length=4)


class TestRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    panel_a: PanelConfig
    panel_b: PanelConfig


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.0.3",
        "roles": list(SUPPORTED_ROLES),
        "models": list(SUPPORTED_MODELS),
    }


@app.post("/api/test")
async def test(request: TestRequest) -> dict:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        return await run_comparison(
            message,
            [agent.model_dump() for agent in request.panel_a.agents],
            [agent.model_dump() for agent in request.panel_b.agents],
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"CREW comparison failed: {exc}",
        ) from exc


@app.post("/api/test-stream")
async def test_stream(request: TestRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    panel_a = [agent.model_dump() for agent in request.panel_a.agents]
    panel_b = [agent.model_dump() for agent in request.panel_b.agents]

    async def stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def progress(event: dict) -> None:
            await queue.put({"type": "progress", **event})

        async def runner() -> None:
            try:
                result = await run_comparison(
                    message,
                    panel_a,
                    panel_b,
                    progress=progress,
                )
                await queue.put({"type": "result", "data": result})
            except Exception as exc:
                await queue.put({"type": "error", "message": str(exc)})

        task = asyncio.create_task(runner())

        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"result", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )