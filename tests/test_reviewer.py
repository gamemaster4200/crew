import unittest
from unittest.mock import AsyncMock, patch

import reviewer


def structured(data, model="gpt-5.6-sol"):
    return {
        "data": data,
        "raw": "{}",
        "model": model,
        "latency_ms": 10,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    }


class ReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_uses_three_sol_calls(self):
        advocate = structured({
            "candidate_a": "A good",
            "candidate_b": "B good",
        })
        adversary = structured({
            "candidate_a": "A bad",
            "candidate_b": "B bad",
        })
        judge = structured({
            "candidate_a_scores": {
                "correctness": 9,
                "completeness": 9,
                "robustness": 8,
                "relevance": 9,
                "actionability": 9,
            },
            "candidate_b_scores": {
                "correctness": 7,
                "completeness": 7,
                "robustness": 7,
                "relevance": 8,
                "actionability": 7,
            },
            "preferred": "A",
            "confidence": "high",
            "verdict": "A is stronger.",
        })

        with patch(
            "reviewer.ask_model_json",
            new=AsyncMock(
                side_effect=[advocate, adversary, judge]
            ),
        ) as mocked, patch(
            "reviewer.random.SystemRandom.shuffle",
            return_value=None,
        ):
            result = await reviewer.review_candidates(
                "task",
                "panel a answer",
                "panel b answer",
            )

        self.assertEqual(mocked.await_count, 3)
        self.assertEqual(result["metrics"]["calls"], 3)
        self.assertEqual(result["preferred_side"], "panel_a")
        self.assertEqual(result["panel_a_view"]["advocate"], "A good")
        self.assertEqual(result["panel_a_view"]["adversary"], "A bad")


if __name__ == "__main__":
    unittest.main()