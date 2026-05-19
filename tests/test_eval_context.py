import unittest

from evaluation.run_eval import _format_context


class TestEvalContext(unittest.TestCase):
    def test_format_context_includes_retrieved_content(self):
        context = _format_context(
            [
                {
                    "subject": "math",
                    "grade": "M4",
                    "topic": "Sets",
                    "section": "Definition",
                    "url": "local://sets",
                    "content": "A set is a well-defined collection of distinct objects.",
                }
            ]
        )

        self.assertIn("math|M4 Sets > Definition", context)
        self.assertIn("URL: local://sets", context)
        self.assertIn("Content:", context)
        self.assertIn("well-defined collection", context)


if __name__ == "__main__":
    unittest.main()
