import unittest
from unittest.mock import AsyncMock, patch

import evaluator


def branch(mode: str, answer: str, calls: int) -> dict:
    return {
        "run_id": f"{mode}-run",
        "mode": mode,
        "task": "task",
        "stages": [],
        "metrics": {
            "calls": calls,
            "latency_ms": 10,
            "input_tokens": 100,
            "output_tokens": 50,
            "reasoning_tokens": 5,
            "total_tokens": 150,
        },
        "final_answer": answer,
        "run_file": "",
    }


def evaluation() -> dict:
    scores = {
        "correctness": 8,
        "completeness": 8,
        "robustness": 8,
        "relevance": 8,
        "actionability": 8,
    }

    return {
        "model": "gpt-5.6-sol",
        "latency_ms": 20,
        "usage": {
            "input_tokens": 200,
            "output_tokens": 100,
            "total_tokens": 300,
            "output_tokens_details": {
                "reasoning_tokens": 20,
            },
        },
        "metrics": {
            "calls": 1,
            "latency_ms": 20,
            "input_tokens": 200,
            "output_tokens": 100,
            "reasoning_tokens": 20,
            "total_tokens": 300,
        },
        "blind_mapping": {
            "A": "crew",
            "B": "single",
        },
        "preferred_blind": "A",
        "preferred_side": "crew",
        "preferred_label": "4x Luna Crew",
        "confidence": "medium",
        "verdict": "Crew is stronger on the task.",
        "candidate_a": {
            "advocate": "Strong.",
            "adversary": "Long.",
            "scores": scores,
        },
        "candidate_b": {
            "advocate": "Concise.",
            "adversary": "Misses edge cases.",
            "scores": scores,
        },
    }


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_test_combines_both_branches_and_evaluator(self):
        with patch(
            "evaluator.run_crew",
            new=AsyncMock(
                return_value=branch("crew", "crew answer", 4)
            ),
        ), patch(
            "evaluator.run_single",
            new=AsyncMock(
                return_value=branch("single", "single answer", 1)
            ),
        ), patch(
            "evaluator.evaluate_candidates",
            new=AsyncMock(return_value=evaluation()),
        ), patch(
            "evaluator.save_run",
            return_value="runs/test.json",
        ):
            result = await evaluator.run_test("task")

        self.assertEqual(result["mode"], "test")
        self.assertEqual(result["crew"]["final_answer"], "crew answer")
        self.assertEqual(result["single"]["final_answer"], "single answer")
        self.assertEqual(
            result["evaluation"]["preferred_side"],
            "crew",
        )
        self.assertEqual(result["test_metrics"]["total_calls"], 6)
        self.assertEqual(result["run_file"], "runs/test.json")


if __name__ == "__main__":
    unittest.main()