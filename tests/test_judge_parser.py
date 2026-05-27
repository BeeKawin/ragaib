import unittest

from evaluation.judge import parse_judge_response


class TestJudgeParser(unittest.TestCase):
    def test_parse_valid_json(self):
        raw = (
            '{"correctness":5,"groundedness":4,"completeness":4,"clarity":5,'
            '"type_alignment":5,"target_answer_type":"exam-prep",'
            '"detected_answer_type":"exam-prep","overall_band":9,'
            '"rationale":"Good answer"}'
        )
        result = parse_judge_response(raw)
        self.assertEqual(result.correctness, 5)
        self.assertEqual(result.groundedness, 4)
        self.assertEqual(result.clarity, 5)
        self.assertEqual(result.type_alignment, 5)
        self.assertEqual(result.target_answer_type, "exam-prep")

    def test_parse_reject_invalid_score(self):
        raw = (
            '{"correctness":0,"groundedness":4,"completeness":4,"clarity":5,'
            '"type_alignment":5,"rationale":"Bad"}'
        )
        with self.assertRaises(ValueError):
            parse_judge_response(raw)

    def test_parse_legacy_safety_as_type_alignment(self):
        raw = (
            '{"correctness":5,"groundedness":4,"completeness":4,"clarity":5,'
            '"safety":3,"rationale":"Legacy answer"}'
        )
        result = parse_judge_response(raw)
        self.assertEqual(result.type_alignment, 3)
        self.assertEqual(result.target_answer_type, "general")


if __name__ == "__main__":
    unittest.main()
