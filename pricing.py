from typing import Any

PRICE_SNAPSHOT_DATE = "2026-09-23"
LONG_CONTEXT_THRESHOLD = 272_000

MODEL_PRICING_USD_PER_MTOK = {
    "gpt-5.6-luna": {
        "short": {
            "input": 0.20,
            "cached_input": 0.02,
            "output": 1.20,
        },
        "long": {
            "input": 0.40,
            "cached_input": 0.04,
            "output": 1.80,
        },
    },
    "gpt-5.6-sol": {
        "short": {
            "input": 4.00,
            "cached_input": 0.40,
            "output": 20.00,
        },
        "long": {
            "input": 8.00,
            "cached_input": 0.80,
            "output": 30.00,
        },
    },
    "gpt-5.6": {
        "short": {
            "input": 4.00,
            "cached_input": 0.40,
            "output": 20.00,
        },
        "long": {
            "input": 8.00,
            "cached_input": 0.80,
            "output": 30.00,
        },
    },
}


def _canonical_model(model: str) -> str:
    model = model.strip()
    if model in MODEL_PRICING_USD_PER_MTOK:
        return model

    for known in ("gpt-5.6-luna", "gpt-5.6-sol"):
        if model.startswith(known):
            return known

    return model


def price_snapshot() -> dict[str, Any]:
    return {
        "date": PRICE_SNAPSHOT_DATE,
        "currency": "USD",
        "unit": "per_1m_tokens",
        "long_context_threshold_input_tokens": LONG_CONTEXT_THRESHOLD,
        "models": MODEL_PRICING_USD_PER_MTOK,
    }


def cost_for_call(
    model: str,
    usage: dict[str, Any] | None,
) -> float | None:
    if usage is None:
        return None

    canonical = _canonical_model(model)
    pricing = MODEL_PRICING_USD_PER_MTOK.get(canonical)
    if pricing is None:
        return None

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    input_details = usage.get("input_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens", 0) or 0)
    cached_tokens = min(cached_tokens, input_tokens)
    uncached_tokens = input_tokens - cached_tokens

    tier = "long" if input_tokens > LONG_CONTEXT_THRESHOLD else "short"
    rates = pricing[tier]

    usd = (
        uncached_tokens * rates["input"]
        + cached_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000

    return round(usd, 10)