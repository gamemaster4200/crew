import asyncio
import random
import time
from typing import Any

from crew_workflow import (
    new_run_id,
    run_crew,
    run_single,
    save_run,
    summarize_stages,
)
from openai_client import ask_model_json, get_evaluator_model

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
        "candidate_a",
        "candidate_b",
        "preferred",
        "confidence",
        "verdict",
    ],
}

EVALUATOR_PROMPT = """\
You are an independent evaluator comparing two anonymous candidate answers to
the same user task.

IMPORTANT:
- You do not know how either candidate was produced.
- Do not infer quality from length, polish, verbosity, or apparent complexity.
- Judge only how well each candidate solves the original task.
- Candidate order is randomized.
- Do not assume consensus or multiple agents imply correctness.

Act as two opposing poles before judging:

ADVOCATE POLE:
For EACH candidate, give the strongest concise case for why that candidate is
good and useful.

ADVERSARY POLE:
For EACH candidate, give the strongest concise case against it: errors,
omissions, unsupported assumptions, fragility, or wasted complexity.

Then score EACH candidate from 0 to 10 on:
- correctness
- completeness
- robustness
- relevance
- actionability

Finally choose A, B, or TIE.

Keep advocate, adversary, and verdict VERY concise.
The verdict should be at most 3 short sentences.

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


def _evaluation_usage_metrics(stage: dict[str, Any]) -> dict[str, int]:
    return summarize_stages(
        [
            {
                "latency_ms": stage["latency_ms"],
                "usage": stage.get("usage"),
            }
        ]
    )


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
) -> dict[str, Any]:
    candidates = [
        ("crew", crew_answer),
        ("single", single_answer),
    ]

    random.SystemRandom().shuffle(candidates)

    mapping = {
        "A": candidates[0][0],
        "B": candidates[1][0],
    }

    prompt = EVALUATOR_PROMPT.format(
        task=task,
        candidate_a=candidates[0][1],
        candidate_b=candidates[1][1],
    )

    model = get_evaluator_model()

    result = await ask_model_json(
        prompt,
        model=model,
        schema_name="crew_comparison_evaluation",
        schema=EVALUATION_SCHEMA,
        reasoning_effort="medium",
    )

    evaluation = result["data"]
    preferred_blind = evaluation["preferred"]

    if preferred_blind == "TIE":
        preferred_side = "tie"
    else:
        preferred_side = mapping[preferred_blind]

    return {
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
        "metrics": _evaluation_usage_metrics(result),
        "blind_mapping": mapping,
        "preferred_blind": preferred_blind,
        "preferred_side": preferred_side,
        "preferred_label": _public_label(preferred_side),
        "confidence": evaluation["confidence"],
        "verdict": evaluation["verdict"],
        "candidate_a": evaluation["candidate_a"],
        "candidate_b": evaluation["candidate_b"],
    }


def _evaluation_by_side(
    evaluation: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    mapping = evaluation["blind_mapping"]

    if mapping["A"] == side:
        return evaluation["candidate_a"]

    return evaluation["candidate_b"]


async def run_test(task: str) -> dict[str, Any]:
    run_id = new_run_id()
    total_started = time.perf_counter()

    solution_started = time.perf_counter()

    crew, single = await asyncio.gather(
        run_crew(task, persist=False),
        run_single(task, persist=False),
    )

    parallel_solution_ms = round(
        (time.perf_counter() - solution_started) * 1000
    )

    evaluation = await evaluate_candidates(
        task,
        crew["final_answer"],
        single["final_answer"],
    )

    total_wall_ms = round(
        (time.perf_counter() - total_started) * 1000
    )

    evaluation["crew_view"] = _evaluation_by_side(
        evaluation,
        "crew",
    )
    evaluation["single_view"] = _evaluation_by_side(
        evaluation,
        "single",
    )

    payload = {
        "run_id": run_id,
        "mode": "test",
        "task": task,
        "crew": crew,
        "single": single,
        "evaluation": evaluation,
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
        },
    }

    payload["run_file"] = save_run(payload)

    return payload