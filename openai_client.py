import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REVIEWER_MODEL = "gpt-5.6-sol"


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put the CREW project API key into .env."
        )
    return AsyncOpenAI(api_key=api_key)


def get_reviewer_model() -> str:
    return (
        os.getenv("OPENAI_EVALUATOR_MODEL", DEFAULT_REVIEWER_MODEL).strip()
        or DEFAULT_REVIEWER_MODEL
    )


def _usage_to_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return None


async def ask_model(
    prompt: str,
    *,
    model: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    client = get_client()
    request: dict[str, Any] = {
        "model": model,
        "input": prompt,
    }
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}

    started = time.perf_counter()
    response = await client.responses.create(**request)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty text response.")

    return {
        "text": text,
        "model": getattr(response, "model", model),
        "latency_ms": elapsed_ms,
        "usage": _usage_to_dict(response),
    }


async def ask_model_json(
    prompt: str,
    *,
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    client = get_client()

    started = time.perf_counter()
    response = await client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    raw = (response.output_text or "").strip()
    if not raw:
        raise RuntimeError("OpenAI returned an empty structured response.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenAI returned invalid structured JSON: {exc}"
        ) from exc

    return {
        "data": data,
        "raw": raw,
        "model": getattr(response, "model", model),
        "latency_ms": elapsed_ms,
        "usage": _usage_to_dict(response),
    }