import unittest
from unittest.mock import AsyncMock, patch

import crew_workflow


def fake_result(text: str) -> dict:
    return {
        "text": text,
        "model": "gpt-5.6-luna",
        "latency_ms": 1000,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 50},
        },
    }


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_metrics_include_cost(self):
        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(return_value=fake_result("single")),
        ):
            result = await crew_workflow.run_single(
                "task",
                persist=False,
            )

        self.assertEqual(result["metrics"]["calls"], 1)
        self.assertGreater(result["metrics"]["cost_usd"], 0)

    async def test_crew_calls_four_roles(self):
        with patch(
            "crew_workflow.ask_model",
            new=AsyncMock(
                side_effect=[
                    fake_result("s"),
                    fake_result("c"),
                    fake_result("i"),
                    fake_result("f"),
                ]
            ),
        ) as mocked:
            result = await crew_workflow.run_crew(
                "task",
                persist=False,
            )

        self.assertEqual(mocked.await_count, 4)
        self.assertEqual(result["final_answer"], "f")
        self.assertEqual(result["metrics"]["calls"], 4)


if __name__ == "__main__":
    unittest.main()