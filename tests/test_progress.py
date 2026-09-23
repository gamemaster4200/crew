import unittest
from unittest.mock import AsyncMock, patch

import crew_workflow


def fake_result(text: str) -> dict:
    return {
        "text": text,
        "model": "gpt-5.6-luna",
        "latency_ms": 10,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    }


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_crew_emits_real_stage_events(self):
        events = []

        async def progress(event):
            events.append(event.copy())

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
        ):
            await crew_workflow.run_crew(
                "task",
                persist=False,
                progress=progress,
            )

        started = [
            e["role"]
            for e in events
            if e["event"] == "stage_started"
        ]
        self.assertEqual(
            started,
            ["Solver", "Critic", "Improver", "Integrator"],
        )


if __name__ == "__main__":
    unittest.main()