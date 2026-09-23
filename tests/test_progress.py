import unittest
from unittest.mock import AsyncMock, patch

import panel_engine


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_panel_progress_sequence(self):
        events = []

        async def progress(event):
            events.append(event.copy())

        response = {
            "text": "answer",
            "model": "gpt-5.6-luna",
            "latency_ms": 10,
            "usage": None,
        }

        with patch(
            "panel_engine.ask_model",
            new=AsyncMock(return_value=response),
        ):
            await panel_engine.run_panel(
                "task",
                "b",
                [{"role": "Solver", "model": "gpt-5.6-luna"}],
                progress=progress,
            )

        self.assertEqual(
            [e["event"] for e in events],
            [
                "panel_started",
                "stage_started",
                "stage_done",
                "panel_done",
            ],
        )


if __name__ == "__main__":
    unittest.main()