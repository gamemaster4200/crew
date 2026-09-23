import os

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


async def ask_model(message: str) -> tuple[str, str]:
    client = get_client()
    model = get_model()

    response = await client.responses.create(
        model=model,
        input=message,
    )

    return response.output_text, model