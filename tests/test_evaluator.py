import unittest

from pricing import cost_for_call


class PricingTests(unittest.TestCase):
    def test_luna_short_context_price(self):
        usage = {
            "input_tokens": 1_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
        }
        self.assertEqual(
            cost_for_call("gpt-5.6-luna", usage),
            0.0002,
        )

    def test_luna_long_context_price(self):
        usage = {
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
        }
        self.assertEqual(
            cost_for_call("gpt-5.6-luna", usage),
            0.4,
        )

    def test_sol_output_price(self):
        usage = {
            "input_tokens": 0,
            "output_tokens": 1_000_000,
        }
        self.assertEqual(
            cost_for_call("gpt-5.6-sol", usage),
            20.0,
        )

    def test_cached_input_price(self):
        usage = {
            "input_tokens": 1_000,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 1_000},
        }
        self.assertEqual(
            cost_for_call("gpt-5.6-luna", usage),
            0.00002,
        )


if __name__ == "__main__":
    unittest.main()