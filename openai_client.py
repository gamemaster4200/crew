import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-5.6-luna"


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


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


async def ask_model(prompt: str) -> dict[str, Any]:
    client = get_client()
    configured_model = get_model()

    started = time.perf_counter()
    response = await client.responses.create(
        model=configured_model,
        input=prompt,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty text response.")

    return {
        "text": text,
        "model": getattr(response, "model", configured_model),
        "latency_ms": elapsed_ms,
        "usage": _usage_to_dict(response),
    }