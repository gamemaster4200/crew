import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai_client import ask_model
from pricing import cost_for_call

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

SUPPORTED_ROLES = ("Solver", "Critic", "Improver", "Integrator")
SUPPORTED_MODELS = ("gpt-5.6-luna", "gpt-5.6-sol")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

ROLE_INSTRUCTIONS = {
    "Solver": """\
Solve the task directly.
Before the solution, briefly identify GOAL, KNOWN CONDITIONS, ASSUMPTIONS, and
UNCERTAINTIES. Do not invent missing facts. If one uncertainty genuinely
blocks a reliable answer, ask one focused blocking question.
""",
    "Critic": """\
Act as a rigorous critic. Inspect the previous work for factual or logical
errors, missed requirements, unsupported assumptions, edge cases, regressions,
and unnecessary complexity. Separate serious findings from optional
improvements. If there is no previous work, independently analyze the task and
identify likely failure modes.
""",
    "Improver": """\
Produce a materially improved candidate solution. Use the original task and
previous work. Accept valid criticism, reject weak criticism, repair defects,
and replace the approach when necessary. Do not invent missing facts.
""",
    "Integrator": """\
Produce the final user-facing answer by synthesizing the strongest parts of the
previous work. Resolve conflicts by evidence and reasoning rather than
majority vote. Preserve the user's actual requirements. If there is no previous
work, solve the task directly.
""",
}


async def emit_progress(
    progress: ProgressCallback | None,
    event: dict[str, Any],
) -> None:
    if progress is None:
        return
    result = progress(event)
    if inspect.isawaitable(result):
        await result


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def save_run(payload: dict[str, Any]) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{payload['run_id']}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path.relative_to(ROOT))


def summarize_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "calls": len(calls),
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_complete": True,
    }

    for call in calls:
        result["latency_ms"] += int(call.get("latency_ms", 0) or 0)
        usage = call.get("usage") or {}

        result["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        result["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        result["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

        details = usage.get("output_tokens_details") or {}
        result["reasoning_tokens"] += int(
            details.get("reasoning_tokens", 0) or 0
        )

        call_cost = call.get("cost_usd")
        if call_cost is None:
            result["cost_complete"] = False
        else:
            result["cost_usd"] += float(call_cost)

    result["cost_usd"] = round(result["cost_usd"], 10)
    return result


def validate_agents(agents: list[dict[str, str]]) -> None:
    if not 1 <= len(agents) <= 4:
        raise ValueError("A panel must contain 1 to 4 agents.")

    for agent in agents:
        role = agent["role"]
        model = agent["model"]

        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")


def _history_text(stages: list[dict[str, Any]]) -> str:
    if not stages:
        return "(none)"

    chunks = []
    for index, stage in enumerate(stages, start=1):
        chunks.append(
            f"PREVIOUS STAGE {index} - {stage['role']}:\n"
            f"<<<\n{stage['text']}\n>>>"
        )
    return "\n\n".join(chunks)


def build_prompt(
    *,
    role: str,
    task: str,
    stages: list[dict[str, Any]],
) -> str:
    return f"""\
You are agent role: {role}.

ROLE INSTRUCTIONS:
{ROLE_INSTRUCTIONS[role]}

ORIGINAL USER TASK:
<<<
{task}
>>>

PREVIOUS PANEL WORK:
{_history_text(stages)}

Return the artifact for your role. Keep it useful to the next agent and faithful
to the original task.
"""


async def run_panel(
    task: str,
    panel_id: str,
    agents: list[dict[str, str]],
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    validate_agents(agents)

    await emit_progress(
        progress,
        {
            "event": "panel_started",
            "panel": panel_id,
            "total": len(agents),
        },
    )

    stages: list[dict[str, Any]] = []

    for index, agent in enumerate(agents, start=1):
        role = agent["role"]
        requested_model = agent["model"]

        await emit_progress(
            progress,
            {
                "event": "stage_started",
                "panel": panel_id,
                "index": index,
                "total": len(agents),
                "role": role,
                "model": requested_model,
            },
        )

        response = await ask_model(
            build_prompt(
                role=role,
                task=task,
                stages=stages,
            ),
            model=requested_model,
        )

        stage = {
            "index": index,
            "role": role,
            "requested_model": requested_model,
            **response,
        }
        stage["cost_usd"] = cost_for_call(
            stage["model"],
            stage.get("usage"),
        )
        stages.append(stage)

        await emit_progress(
            progress,
            {
                "event": "stage_done",
                "panel": panel_id,
                "index": index,
                "total": len(agents),
                "role": role,
                "model": stage["model"],
                "latency_ms": stage["latency_ms"],
            },
        )

    payload = {
        "panel": panel_id,
        "config": {"agents": agents},
        "stages": stages,
        "metrics": summarize_calls(stages),
        "final_answer": stages[-1]["text"],
    }

    await emit_progress(
        progress,
        {
            "event": "panel_done",
            "panel": panel_id,
            "result": payload,
        },
    )

    return payload