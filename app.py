from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from crew_workflow import run_crew, run_single
from evaluator import run_test

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"

app = FastAPI(title="CREW", version="0.0.2-test")


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
        "version": "0.0.2-test",
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