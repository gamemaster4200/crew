import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from crew_workflow import run_crew, run_single
from evaluator import run_test

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"

app = FastAPI(title="CREW", version="0.0.2-test-progress")


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    mode: Literal["crew", "single"] = "crew"


class TestRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "0.0.2-test-progress",
    }


@app.post("/api/run")
async def run(request: RunRequest) -> dict:
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message is empty.",
        )

    try:
        if request.mode == "single":
            return await run_single(message)

        return await run_crew(message)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"CREW run failed: {exc}",
        ) from exc


@app.post("/api/test")
async def test(request: TestRequest) -> dict:
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message is empty.",
        )

    try:
        return await run_test(message)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"CREW test failed: {exc}",
        ) from exc


@app.post("/api/test-stream")
async def test_stream(request: TestRequest) -> StreamingResponse:
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message is empty.",
        )

    async def stream():
        queue: asyncio.Queue[dict] = asyncio.Queue()

        async def progress(event: dict) -> None:
            await queue.put(
                {
                    "type": "progress",
                    **event,
                }
            )

        async def runner() -> None:
            try:
                result = await run_test(
                    message,
                    progress=progress,
                )

                await queue.put(
                    {
                        "type": "result",
                        "data": result,
                    }
                )

            except Exception as exc:
                await queue.put(
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                )

        task = asyncio.create_task(runner())

        try:
            while True:
                event = await queue.get()

                yield (
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

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