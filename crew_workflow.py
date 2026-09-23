import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai_client import ask_model

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

ROLES = ("Solver", "Critic", "Improver", "Integrator")

SOLVER_PROMPT = """\
You are the Solver in a fixed four-agent CREW.

Your job:
1. Understand the task.
2. Explicitly identify:
   - GOAL
   - KNOWN CONDITIONS
   - ASSUMPTIONS
   - UNCERTAINTIES
3. Propose a concrete solution.
4. Do not spend most of the answer criticizing yourself.
5. Do not invent missing facts.
6. If an essential ambiguity makes a reliable solution impossible, state one
   targeted BLOCKING QUESTION.

USER TASK:
<<<
{task}
>>>
"""

CRITIC_PROMPT = """\
You are the Critic in a fixed four-agent CREW.

You receive the original user task and the Solver result.
Do not simply rewrite the solution.

Actively search for:
- factual or logical errors;
- missed requirements;
- unsupported assumptions;
- weak reasoning;
- edge cases;
- regressions or unwanted consequences;
- places where clarification is genuinely required.

Separate serious findings from optional improvements.
Do not treat disagreement as proof that the Solver is wrong.

USER TASK:
<<<
{task}
>>>

SOLVER RESULT:
<<<
{solver}
>>>
"""

IMPROVER_PROMPT = """\
You are the Improver in a fixed four-agent CREW.

You receive the original task, the Solver result, and the Critic review.

Your job:
- decide which Critic findings are valid;
- reject Critic findings that are weak or irrelevant;
- repair the Solver result where possible;
- replace the approach if the original solution is fundamentally flawed;
- produce one improved candidate answer.

At the end, briefly list:
ACCEPTED CRITIC FINDINGS
REJECTED CRITIC FINDINGS

Do not invent missing facts.

USER TASK:
<<<
{task}
>>>

SOLVER RESULT:
<<<
{solver}
>>>

CRITIC REVIEW:
<<<
{critic}
>>>
"""

INTEGRATOR_PROMPT = """\
You are the Integrator in a fixed four-agent CREW.

You receive:
- the original user task;
- the Solver result;
- the Critic review;
- the Improver result.

Produce the final user-facing answer.

Rules:
- Resolve conflicts by evidence and reasoning, not by majority vote.
- Preserve the actual user requirements.
- Do not invent facts to close gaps.
- If an essential uncertainty remains, ask a focused clarification question.
- Do not discuss the internal CREW process unless it is directly useful.
- Return a clean final answer, not a review of the other agents.

USER TASK:
<<<
{task}
>>>

SOLVER RESULT:
<<<
{solver}
>>>

CRITIC REVIEW:
<<<
{critic}
>>>

IMPROVER RESULT:
<<<
{improver}
>>>
"""

SINGLE_PROMPT = """\
You are a capable general assistant acting as the single-agent baseline for an
experiment.

Solve the user task directly and accurately.
Do not invent missing facts.
If an essential ambiguity blocks a reliable answer, ask one targeted
clarification question.

USER TASK:
<<<
{task}
>>>
"""


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


def summarize_stages(stages: list[dict[str, Any]]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0
    latency_ms = 0

    for stage in stages:
        latency_ms += int(stage.get("latency_ms", 0) or 0)
        usage = stage.get("usage") or {}

        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)

        details = usage.get("output_tokens_details") or {}
        reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)

    return {
        "calls": len(stages),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


async def _stage(role: str, prompt: str) -> dict[str, Any]:
    result = await ask_model(prompt)

    return {
        "role": role,
        **result,
    }


async def run_single(
    task: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    run_id = new_run_id()
    stage = await _stage("Single", SINGLE_PROMPT.format(task=task))
    stages = [stage]

    payload = {
        "run_id": run_id,
        "mode": "single",
        "task": task,
        "stages": stages,
        "metrics": summarize_stages(stages),
        "final_answer": stage["text"],
    }

    if persist:
        payload["run_file"] = save_run(payload)
    else:
        payload["run_file"] = ""

    return payload


async def run_crew(
    task: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    run_id = new_run_id()

    solver = await _stage(
        "Solver",
        SOLVER_PROMPT.format(task=task),
    )

    critic = await _stage(
        "Critic",
        CRITIC_PROMPT.format(
            task=task,
            solver=solver["text"],
        ),
    )

    improver = await _stage(
        "Improver",
        IMPROVER_PROMPT.format(
            task=task,
            solver=solver["text"],
            critic=critic["text"],
        ),
    )

    integrator = await _stage(
        "Integrator",
        INTEGRATOR_PROMPT.format(
            task=task,
            solver=solver["text"],
            critic=critic["text"],
            improver=improver["text"],
        ),
    )

    stages = [solver, critic, improver, integrator]

    payload = {
        "run_id": run_id,
        "mode": "crew",
        "task": task,
        "roles": list(ROLES),
        "stages": stages,
        "metrics": summarize_stages(stages),
        "final_answer": integrator["text"],
    }

    if persist:
        payload["run_file"] = save_run(payload)
    else:
        payload["run_file"] = ""

    return payload