import asyncio
import random
import time
from typing import Any

from crew_workflow import (
    ProgressCallback,
    emit_progress,
    new_run_id,
    run_crew,
    run_single,
    save_run,
    summarize_stages,
)
from openai_client import ask_model_json, get_evaluator_model
from pricing import cost_for_call, price_snapshot

SCORE_NAMES = (
    "correctness",
    "completeness",
    "robustness",
    "relevance",
    "actionability",
)

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

CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "advocate": {"type": "string"},
        "adversary": {"type": "string"},
        "scores": SCORE_SCHEMA,
    },
    "required": ["advocate", "adversary", "scores"],
}

EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_a": CANDIDATE_SCHEMA,
        "candidate_b": CANDIDATE_SCHEMA,
        "preferred": {"type": "string", "enum": ["A", "B", "TIE"]},
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "verdict": {"type": "string"},
    },
    "required": [
        "candidate_a",
        "candidate_b",
        "preferred",
        "confidence",
        "verdict",
    ],
}

EVALUATOR_PROMPT = """\
You are an independent evaluator comparing two anonymous candidate answers.

Do not infer quality from length, polish, verbosity, or apparent complexity.
Judge only how well each candidate solves the original task.
Candidate order is randomized.
Do not assume multiple agents imply correctness.

For EACH candidate:
- give the strongest concise ADVOCATE case;
- give the strongest concise ADVERSARY case;
- score 0-10 on correctness, completeness, robustness, relevance, actionability.

Finally choose A, B, or TIE and give a verdict of at most 3 short sentences.

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


def _public_label(side: str) -> str:
    if side == "crew":
        return "4x Luna Crew"
    if side == "single":
        return "1x Luna"
    return "Tie"


async def evaluate_candidates(
    task: str,
    crew_answer: str,
    single_answer: str,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    candidates = [("crew", crew_answer), ("single", single_answer)]
    random.SystemRandom().shuffle(candidates)

    mapping = {"A": candidates[0][0], "B": candidates[1][0]}

    model = get_evaluator_model()
    await emit_progress(
        progress,
        {"event": "evaluator_started", "model": model},
    )

    result = await ask_model_json(
        EVALUATOR_PROMPT.format(
            task=task,
            candidate_a=candidates[0][1],
            candidate_b=candidates[1][1],
        ),
        model=model,
        schema_name="crew_comparison_evaluation",
        schema=EVALUATION_SCHEMA,
        reasoning_effort="medium",
    )

    data = result["data"]
    preferred_blind = data["preferred"]
    preferred_side = (
        "tie" if preferred_blind == "TIE" else mapping[preferred_blind]
    )

    stage = {
        "role": "Evaluator",
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
    }
    stage["cost_usd"] = cost_for_call(stage["model"], stage.get("usage"))

    payload = {
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
        "cost_usd": stage["cost_usd"],
        "metrics": summarize_stages([stage]),
        "blind_mapping": mapping,
        "preferred_blind": preferred_blind,
        "preferred_side": preferred_side,
        "preferred_label": _public_label(preferred_side),
        "confidence": data["confidence"],
        "verdict": data["verdict"],
        "candidate_a": data["candidate_a"],
        "candidate_b": data["candidate_b"],
    }

    await emit_progress(
        progress,
        {
            "event": "evaluator_done",
            "model": payload["model"],
            "latency_ms": payload["latency_ms"],
        },
    )
    return payload


def _evaluation_by_side(
    evaluation: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    return (
        evaluation["candidate_a"]
        if evaluation["blind_mapping"]["A"] == side
        else evaluation["candidate_b"]
    )


async def run_test(
    task: str,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    total_started = time.perf_counter()

    await emit_progress(
        progress,
        {"event": "test_started", "run_id": run_id},
    )

    solution_started = time.perf_counter()
    crew, single = await asyncio.gather(
        run_crew(task, persist=False, progress=progress),
        run_single(task, persist=False, progress=progress),
    )
    parallel_solution_ms = round(
        (time.perf_counter() - solution_started) * 1000
    )

    evaluation = await evaluate_candidates(
        task,
        crew["final_answer"],
        single["final_answer"],
        progress=progress,
    )

    evaluation["crew_view"] = _evaluation_by_side(evaluation, "crew")
    evaluation["single_view"] = _evaluation_by_side(evaluation, "single")

    crew_cost = crew["metrics"]["cost_usd"]
    single_cost = single["metrics"]["cost_usd"]
    delta_cost = crew_cost - single_cost
    cost_ratio = (
        crew_cost / single_cost
        if single_cost > 0
        else None
    )

    total_wall_ms = round(
        (time.perf_counter() - total_started) * 1000
    )

    payload = {
        "run_id": run_id,
        "mode": "test",
        "task": task,
        "crew": crew,
        "single": single,
        "evaluation": evaluation,
        "pricing": price_snapshot(),
        "test_metrics": {
            "parallel_solution_ms": parallel_solution_ms,
            "evaluator_ms": evaluation["latency_ms"],
            "total_wall_ms": total_wall_ms,
            "total_calls": (
                crew["metrics"]["calls"]
                + single["metrics"]["calls"]
                + 1
            ),
            "total_input_tokens": (
                crew["metrics"]["input_tokens"]
                + single["metrics"]["input_tokens"]
                + evaluation["metrics"]["input_tokens"]
            ),
            "total_output_tokens": (
                crew["metrics"]["output_tokens"]
                + single["metrics"]["output_tokens"]
                + evaluation["metrics"]["output_tokens"]
            ),
            "total_reasoning_tokens": (
                crew["metrics"]["reasoning_tokens"]
                + single["metrics"]["reasoning_tokens"]
                + evaluation["metrics"]["reasoning_tokens"]
            ),
            "total_tokens": (
                crew["metrics"]["total_tokens"]
                + single["metrics"]["total_tokens"]
                + evaluation["metrics"]["total_tokens"]
            ),
            "crew_cost_usd": crew_cost,
            "single_cost_usd": single_cost,
            "solution_cost_delta_usd": delta_cost,
            "solution_cost_ratio": cost_ratio,
            "evaluator_cost_usd": evaluation["metrics"]["cost_usd"],
            "total_cost_usd": round(
                crew_cost
                + single_cost
                + evaluation["metrics"]["cost_usd"],
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