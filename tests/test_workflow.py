import unittest
from unittest.mock import AsyncMock, patch

import crew_workflow


def fake_result(text: str) -> dict:
    return {
        "text": text,
        "model": "gpt-5.6-luna",
        "latency_ms": 1,
        "usage": None,
    }


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_calls_model_once(self):
        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(return_value=fake_result("single answer")),
        ) as mocked, patch(
            "crew_workflow._save_run",
            return_value="runs/fake.json",
        ):
            result = await crew_workflow.run_single("task")

        self.assertEqual(result["mode"], "single")
        self.assertEqual(result["final_answer"], "single answer")
        self.assertEqual([s["role"] for s in result["stages"]], ["Single"])
        self.assertEqual(mocked.await_count, 1)

    async def test_crew_calls_four_roles_in_order(self):
        responses = [
            fake_result("solver answer"),
            fake_result("critic answer"),
            fake_result("improver answer"),
            fake_result("integrator answer"),
        ]

        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(side_effect=responses),
        ) as mocked, patch(
            "crew_workflow._save_run",
            return_value="runs/fake.json",
        ):
            result = await crew_workflow.run_crew("task")

        self.assertEqual(result["mode"], "crew")
        self.assertEqual(
            [s["role"] for s in result["stages"]],
            ["Solver", "Critic", "Improver", "Integrator"],
        )
        self.assertEqual(result["final_answer"], "integrator answer")
        self.assertEqual(mocked.await_count, 4)


if __name__ == "__main__":
    unittest.main()