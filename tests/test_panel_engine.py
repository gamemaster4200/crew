import unittest
from unittest.mock import AsyncMock, patch

import panel_engine


def fake(text: str, model: str) -> dict:
    return {
        "text": text,
        "model": model,
        "latency_ms": 10,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    }


class PanelEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_panel_uses_configured_roles_and_models(self):
        agents = [
            {"role": "Solver", "model": "gpt-5.6-luna"},
            {"role": "Critic", "model": "gpt-5.6-sol"},
            {"role": "Integrator", "model": "gpt-5.6-luna"},
        ]

        async def side_effect(prompt, *, model, reasoning_effort=None):
            return fake(model, model)

        with patch(
            "panel_engine.ask_model",
            new=AsyncMock(side_effect=side_effect),
        ) as mocked:
            result = await panel_engine.run_panel(
                "task",
                "a",
                agents,
            )

        self.assertEqual(mocked.await_count, 3)
        self.assertEqual(
            [s["role"] for s in result["stages"]],
            ["Solver", "Critic", "Integrator"],
        )
        self.assertEqual(
            [s["requested_model"] for s in result["stages"]],
            [
                "gpt-5.6-luna",
                "gpt-5.6-sol",
                "gpt-5.6-luna",
            ],
        )
        self.assertEqual(result["final_answer"], "gpt-5.6-luna")

    async def test_rejects_five_agents(self):
        agents = [
            {"role": "Solver", "model": "gpt-5.6-luna"}
            for _ in range(5)
        ]
        with self.assertRaises(ValueError):
            await panel_engine.run_panel("task", "a", agents)


if __name__ == "__main__":
    unittest.main()