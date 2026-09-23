import unittest
from unittest.mock import AsyncMock, patch

import crew_workflow
import evaluator


def fake_model_result(text: str) -> dict:
    return {
        "text": text,
        "model": "gpt-5.6-luna",
        "latency_ms": 10,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {
                "reasoning_tokens": 1,
            },
        },
    }


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_crew_emits_stage_progress(self):
        events = []

        async def progress(event):
            events.append(event.copy())

        results = [
            fake_model_result("solver"),
            fake_model_result("critic"),
            fake_model_result("improver"),
            fake_model_result("integrator"),
        ]

        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(side_effect=results),
        ):
            result = await crew_workflow.run_crew(
                "task",
                persist=False,
                progress=progress,
            )

        self.assertEqual(result["final_answer"], "integrator")

        started = [
            event["role"]
            for event in events
            if event["event"] == "stage_started"
        ]
        done = [
            event["role"]
            for event in events
            if event["event"] == "stage_done"
        ]

        self.assertEqual(
            started,
            ["Solver", "Critic", "Improver", "Integrator"],
        )
        self.assertEqual(done, started)
        self.assertEqual(events[0]["event"], "branch_started")
        self.assertEqual(events[-1]["event"], "branch_done")

    async def test_single_emits_progress(self):
        events = []

        async def progress(event):
            events.append(event.copy())

        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(return_value=fake_model_result("single")),
        ):
            result = await crew_workflow.run_single(
                "task",
                persist=False,
                progress=progress,
            )

        self.assertEqual(result["final_answer"], "single")
        self.assertEqual(
            [event["event"] for event in events],
            [
                "branch_started",
                "stage_started",
                "stage_done",
                "branch_done",
            ],
        )

    async def test_evaluator_emits_start_and_done(self):
        events = []

        async def progress(event):
            events.append(event.copy())

        fake_eval = {
            "data": {
                "candidate_a": {
                    "advocate": "a",
                    "adversary": "b",
                    "scores": {
                        "correctness": 8,
                        "completeness": 8,
                        "robustness": 8,
                        "relevance": 8,
                        "actionability": 8,
                    },
                },
                "candidate_b": {
                    "advocate": "c",
                    "adversary": "d",
                    "scores": {
                        "correctness": 7,
                        "completeness": 7,
                        "robustness": 7,
                        "relevance": 7,
                        "actionability": 7,
                    },
                },
                "preferred": "A",
                "confidence": "medium",
                "verdict": "A is better.",
            },
            "raw": "{}",
            "model": "gpt-5.6-sol",
            "latency_ms": 20,
            "usage": None,
        }

        with patch(
            "evaluator.ask_model_json",
            new=AsyncMock(return_value=fake_eval),
        ), patch(
            "evaluator.random.SystemRandom.shuffle",
            return_value=None,
        ):
            result = await evaluator.evaluate_candidates(
                "task",
                "crew",
                "single",
                progress=progress,
            )

        self.assertEqual(result["preferred_side"], "crew")
        self.assertEqual(
            [event["event"] for event in events],
            ["evaluator_started", "evaluator_done"],
        )


if __name__ == "__main__":
    unittest.main()