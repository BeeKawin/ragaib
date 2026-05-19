import unittest

from evaluation.judge import JudgeResult
from evaluation.scoring import weighted_score


class TestWeightedScore(unittest.TestCase):
    def test_weighted_score_range(self):
        s = JudgeResult(
            correctness=5,
            groundedness=5,
            completeness=5,
            clarity=5,
            safety=5,
            rationale="",
        )
        self.assertEqual(weighted_score(s), 100.0)

    def test_weighted_score_computation(self):
        s = JudgeResult(
            correctness=3,
            groundedness=4,
            completeness=5,
            clarity=2,
            safety=1,
            rationale="",
        )
        score = weighted_score(s)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


if __name__ == "__main__":
    unittest.main()
