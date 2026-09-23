import asyncio
import random
import time
from typing import Any

from openai_client import ask_model_json, get_reviewer_model
from panel_engine import (
    ProgressCallback,
    emit_progress,
    new_run_id,
    run_panel,
    save_run,
    summarize_calls,
)
from pricing import cost_for_call, price_snapshot

SCORE_NAMES = (
    "correctness",
    "completeness",
    "robustness",
    "relevance",
    "actionability",
)

PAIR_TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_a": {"type": "string"},
        "candidate_b": {"type": "string"},
    },
    "required": ["candidate_a", "candidate_b"],
}

SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correctness": {"type": "integer"},
        "completeness": {"type": "integer"},
        "robustness": {"type": "integer"},
        "relevance": {"type": "integer"},
        "actionability": {"type": "integer"},
    },
    "required": list(SCORE_NAMES),
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_a_scores": SCORE_SCHEMA,
        "candidate_b_scores": SCORE_SCHEMA,
        "preferred": {
            "type": "string",
            "enum": ["A", "B", "TIE"],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "verdict": {"type": "string"},
    },
    "required": [
        "candidate_a_scores",
        "candidate_b_scores",
        "preferred",
        "confidence",
        "verdict",
    ],
}

ADVOCATE_PROMPT = """\
You are the ADVOCATE in a blind comparison.

Build the strongest concise case FOR EACH candidate.
Do not choose a winner.
Do not infer anything from length or style.
Focus on correctness, completeness, robustness, relevance, and actionability.
Inside each returned candidate field, do not call it A or B. Write only the
substantive case; the UI already knows which panel that field belongs to.

USER TASK:
<<<
{task}
>>>

CANDIDATE A:
<<<
{candidate_a}
>>>

CANDIDATE B:
<<<
{candidate_b}
>>>
"""

ADVERSARY_PROMPT = """\
You are the ADVERSARY in a blind comparison.

Build the strongest concise case AGAINST EACH candidate.
Search aggressively for errors, omissions, unsupported assumptions, fragility,
missed edge cases, irrelevant complexity, and violations of the user request.
Do not choose a winner.
Inside each returned candidate field, do not call it A or B. Write only the
substantive case; the UI already knows which panel that field belongs to.

USER TASK:
<<<
{task}
>>>

CANDIDATE A:
<<<
{candidate_a}
>>>

CANDIDATE B:
<<<
{candidate_b}
>>>
"""

JUDGE_PROMPT = """\
You are the final JUDGE in a blind comparison.

You receive the original task, both candidate answers, an independent Advocate
report, and an independent Adversary report.

Do not reward length, polish, verbosity, or apparent complexity.
Do not assume the Advocate or Adversary is correct.
Resolve conflicts by evidence and reasoning.

Score EACH candidate 0-10 on:
- correctness
- completeness
- robustness
- relevance
- actionability

Choose A, B, or TIE.
Give a verdict of at most 3 short sentences.
In the verdict text, do not refer to the candidates as A or B and do not repeat
the winner label. Explain only the substantive reason for the judgment. The UI
will translate the blind winner to Panel A or Panel B separately.

USER TASK:
<<<
{task}
>>>

CANDIDATE A:
<<<
{candidate_a}
>>>

CANDIDATE B:
<<<
{candidate_b}
>>>

ADVOCATE REPORT:
Candidate A: {advocate_a}
Candidate B: {advocate_b}

ADVERSARY REPORT:
Candidate A: {adversary_a}
Candidate B: {adversary_b}
"""


def _review_call(
    role: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    call = {
        "role": role,
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
        "data": result["data"],
    }
    call["cost_usd"] = cost_for_call(
        call["model"],
        call.get("usage"),
    )
    return call


async def _run_pole(
    *,
    pole: str,
    prompt: str,
    model: str,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    await emit_progress(
        progress,
        {
            "event": "reviewer_started",
            "reviewer": pole.lower(),
            "model": model,
        },
    )

    result = await ask_model_json(
        prompt,
        model=model,
        schema_name=f"crew_{pole.lower()}_report",
        schema=PAIR_TEXT_SCHEMA,
        reasoning_effort="medium",
    )
    call = _review_call(pole, result)

    await emit_progress(
        progress,
        {
            "event": "reviewer_done",
            "reviewer": pole.lower(),
            "model": call["model"],
            "latency_ms": call["latency_ms"],
        },
    )
    return call


async def review_candidates(
    task: str,
    panel_a_answer: str,
    panel_b_answer: str,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    candidates = [
        ("panel_a", panel_a_answer),
        ("panel_b", panel_b_answer),
    ]
    random.SystemRandom().shuffle(candidates)

    mapping = {
        "A": candidates[0][0],
        "B": candidates[1][0],
    }
    candidate_a = candidates[0][1]
    candidate_b = candidates[1][1]

    model = get_reviewer_model()

    await emit_progress(
        progress,
        {"event": "review_started", "model": model},
    )

    advocate_prompt = ADVOCATE_PROMPT.format(
        task=task,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )
    adversary_prompt = ADVERSARY_PROMPT.format(
        task=task,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    advocate, adversary = await asyncio.gather(
        _run_pole(
            pole="Advocate",
            prompt=advocate_prompt,
            model=model,
            progress=progress,
        ),
        _run_pole(
            pole="Adversary",
            prompt=adversary_prompt,
            model=model,
            progress=progress,
        ),
    )

    await emit_progress(
        progress,
        {
            "event": "judge_started",
            "model": model,
        },
    )

    judge_result = await ask_model_json(
        JUDGE_PROMPT.format(
            task=task,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            advocate_a=advocate["data"]["candidate_a"],
            advocate_b=advocate["data"]["candidate_b"],
            adversary_a=adversary["data"]["candidate_a"],
            adversary_b=adversary["data"]["candidate_b"],
        ),
        model=model,
        schema_name="crew_final_judgment",
        schema=JUDGE_SCHEMA,
        reasoning_effort="medium",
    )
    judge = _review_call("Judge", judge_result)

    await emit_progress(
        progress,
        {
            "event": "judge_done",
            "model": judge["model"],
            "latency_ms": judge["latency_ms"],
        },
    )

    preferred_blind = judge["data"]["preferred"]
    preferred_side = (
        "tie"
        if preferred_blind == "TIE"
        else mapping[preferred_blind]
    )

    def candidate_key(side: str) -> str:
        return "candidate_a" if mapping["A"] == side else "candidate_b"

    def score_key(side: str) -> str:
        return (
            "candidate_a_scores"
            if mapping["A"] == side
            else "candidate_b_scores"
        )

    panel_a_key = candidate_key("panel_a")
    panel_b_key = candidate_key("panel_b")

    calls = [advocate, adversary, judge]

    payload = {
        "model": model,
        "blind_mapping": mapping,
        "calls": calls,
        "metrics": summarize_calls(calls),
        "advocate": advocate,
        "adversary": adversary,
        "judge": judge,
        "preferred_blind": preferred_blind,
        "preferred_side": preferred_side,
        "confidence": judge["data"]["confidence"],
        "verdict": judge["data"]["verdict"],
        "panel_a_view": {
            "advocate": advocate["data"][panel_a_key],
            "adversary": adversary["data"][panel_a_key],
            "scores": judge["data"][score_key("panel_a")],
        },
        "panel_b_view": {
            "advocate": advocate["data"][panel_b_key],
            "adversary": adversary["data"][panel_b_key],
            "scores": judge["data"][score_key("panel_b")],
        },
    }

    await emit_progress(
        progress,
        {"event": "review_done", "model": model},
    )

    return payload


async def run_comparison(
    task: str,
    panel_a_agents: list[dict[str, str]],
    panel_b_agents: list[dict[str, str]],
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    started = time.perf_counter()

    await emit_progress(
        progress,
        {"event": "test_started", "run_id": run_id},
    )

    panels_started = time.perf_counter()
    panel_a, panel_b = await asyncio.gather(
        run_panel(
            task,
            "a",
            panel_a_agents,
            progress=progress,
        ),
        run_panel(
            task,
            "b",
            panel_b_agents,
            progress=progress,
        ),
    )
    panel_wall_ms = round(
        (time.perf_counter() - panels_started) * 1000
    )

    review_started = time.perf_counter()
    review = await review_candidates(
        task,
        panel_a["final_answer"],
        panel_b["final_answer"],
        progress=progress,
    )
    review_wall_ms = round(
        (time.perf_counter() - review_started) * 1000
    )

    total_wall_ms = round(
        (time.perf_counter() - started) * 1000
    )

    cost_a = panel_a["metrics"]["cost_usd"]
    cost_b = panel_b["metrics"]["cost_usd"]
    delta = round(cost_a - cost_b, 10)
    ratio = cost_a / cost_b if cost_b > 0 else None

    payload = {
        "run_id": run_id,
        "mode": "comparison",
        "task": task,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "review": review,
        "pricing": price_snapshot(),
        "test_metrics": {
            "panel_wall_ms": panel_wall_ms,
            "review_wall_ms": review_wall_ms,
            "total_wall_ms": total_wall_ms,
            "total_calls": (
                panel_a["metrics"]["calls"]
                + panel_b["metrics"]["calls"]
                + review["metrics"]["calls"]
            ),
            "total_tokens": (
                panel_a["metrics"]["total_tokens"]
                + panel_b["metrics"]["total_tokens"]
                + review["metrics"]["total_tokens"]
            ),
            "panel_a_cost_usd": cost_a,
            "panel_b_cost_usd": cost_b,
            "solution_cost_delta_usd": delta,
            "solution_cost_ratio": ratio,
            "review_cost_usd": review["metrics"]["cost_usd"],
            "total_cost_usd": round(
                cost_a + cost_b + review["metrics"]["cost_usd"],
                10,
            ),
        },
    }

    payload["run_file"] = save_run(payload)

    await emit_progress(
        progress,
        {"event": "test_done", "run_id": run_id},
    )
    return payload