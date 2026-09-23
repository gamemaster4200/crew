import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EVALUATOR_MODEL = "gpt-5.6-sol"


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_evaluator_model() -> str:
    return (
        os.getenv("OPENAI_EVALUATOR_MODEL", DEFAULT_EVALUATOR_MODEL).strip()
        or DEFAULT_EVALUATOR_MODEL
    )


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Put the CREW project API key into .env."
        )

    return AsyncOpenAI(api_key=api_key)


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
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    client = get_client()
    selected_model = model or get_model()

    request: dict[str, Any] = {
        "model": selected_model,
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
        "model": getattr(response, "model", selected_model),
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
        raise RuntimeError("Evaluator returned an empty response.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Evaluator returned invalid JSON: {exc}"
        ) from exc

    return {
        "data": data,
        "raw": raw,
        "model": getattr(response, "model", model),
        "latency_ms": elapsed_ms,
        "usage": _usage_to_dict(response),
    }