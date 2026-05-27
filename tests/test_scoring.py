import unittest

from evaluation.judge import JudgeResult, calculate_overall_band


class TestOverallBand(unittest.TestCase):
    def test_overall_band_range(self):
        self.assertEqual(calculate_overall_band(5, 5, 5, 5, 5), 10)

    def test_overall_band_applies_caps(self):
        score = JudgeResult(
            correctness=3,
            groundedness=4,
            completeness=5,
            clarity=2,
            type_alignment=1,
            target_answer_type="general",
            detected_answer_type="quick-answer",
            overall_band=calculate_overall_band(3, 4, 5, 2, 1),
            rationale="",
        )
        self.assertGreaterEqual(score.overall_band, 1)
        self.assertLessEqual(score.overall_band, 8)


if __name__ == "__main__":
    unittest.main()
