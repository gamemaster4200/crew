from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from crew_workflow import run_crew, run_single

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"

app = FastAPI(title="CREW", version="0.0.2")


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    mode: Literal["crew", "single"] = "crew"


class StageResponse(BaseModel):
    role: str
    text: str
    model: str
    latency_ms: int
    usage: dict | None = None


class RunResponse(BaseModel):
    run_id: str
    mode: str
    final_answer: str
    stages: list[StageResponse]
    run_file: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.0.2"}


@app.post("/api/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    try:
        if request.mode == "single":
            result = await run_single(message)
        else:
            result = await run_crew(message)

        return RunResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"CREW run failed: {exc}",
        ) from exc