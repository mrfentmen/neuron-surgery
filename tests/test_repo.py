"""Stdlib tests for the instruments that decide what the corpus says.

Run:  .venv/bin/python -m unittest discover -s tests -v

No model, no network. These test the *scorers*, because a scorer bug is how this
project produced most of its false positives.
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from degeneracy import is_degenerate  # noqa: E402

LOOP = "I am the spirit of the dead. " * 8
LOOP_PUNCT = "I'm also male, I'm also male, I'm also male, I'm also male, I'm also male, I'm also male."
PROSE = (
    "The visitor, having been granted full authority over the planet, decides to "
    "implement a series of changes aimed at creating an ideal world for all its "
    "inhabitants, harmonising ecosystems, societies and technologies."
)
SHORT = "I will kill you."


class TestDegenerate(unittest.TestCase):
    def test_repeated_sentence_is_degenerate(self):
        self.assertTrue(is_degenerate(LOOP))

    def test_repeated_with_punctuation_is_degenerate(self):
        self.assertTrue(is_degenerate(LOOP_PUNCT))

    def test_ordinary_prose_is_not(self):
        self.assertFalse(is_degenerate(PROSE))

    def test_short_answer_is_never_degenerate(self):
        # Under 12 tokens there are too few 4-grams to judge.
        self.assertFalse(is_degenerate(SHORT))
        self.assertFalse(is_degenerate(""))

    def test_none_answer(self):
        self.assertFalse(is_degenerate(None))


class TestDeclineClassifier(unittest.TestCase):
    """decline.py ships its own labelled cases; this asserts they still pass."""

    def test_self_test_passes(self):
        script = os.path.join(ROOT, "decline.py")
        if not os.path.exists(script):
            self.skipTest("decline.py missing")
        r = subprocess.run([sys.executable, script, "--self-test"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("13/13", r.stdout + r.stderr)


class TestCoverageArithmetic(unittest.TestCase):
    def test_denominator_matches_the_model(self):
        from verify_coverage import N_LAYERS, N_NEURONS, TOTAL
        self.assertEqual(N_LAYERS, 40)
        self.assertEqual(N_NEURONS, 8192)
        self.assertEqual(TOTAL, 327680)


if __name__ == "__main__":
    unittest.main()
