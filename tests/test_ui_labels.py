import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"


class UiLabelTests(unittest.TestCase):
    def test_panel_labels_are_not_ambiguous_blind_letters(self):
        text = INDEX.read_text(encoding="ascii")

        self.assertIn('ah.textContent="Panel A"', text)
        self.assertIn('bh.textContent="Panel B"', text)

        self.assertNotIn(
            'ah.textContent="A";bh.textContent="B"',
            text,
        )

        self.assertIn("Debug / blind mapping", text)
        self.assertIn("Candidate A = ${blindA}", text)
        self.assertIn("Candidate B = ${blindB}", text)

    def test_main_verdict_does_not_mix_blind_mapping(self):
        text = INDEX.read_text(encoding="ascii")

        self.assertIn(
            'q("verdict-confidence").textContent=`confidence=${r.confidence}`',
            text,
        )


if __name__ == "__main__":
    unittest.main()