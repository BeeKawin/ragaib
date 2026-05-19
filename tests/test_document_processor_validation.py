import unittest

from embeddings.document_processor import validate_raw_page


def _valid_raw_page():
    return {
        "subject": "physics",
        "grade": "M4",
        "topic": "Newton's Second Law",
        "url": "local://physics/m4/newton",
        "title": "Newton's Second Law",
        "sections": [
            {
                "heading": "Core idea",
                "content": "Newton's second law states that net force equals mass times acceleration.",
            }
        ],
        "key_terms": ["net force", "mass", "acceleration"],
        "equations": ["F = ma"],
        "examples": [],
        "raw_text": "Newton's second law states that net force equals mass times acceleration.",
    }


class TestDocumentProcessorValidation(unittest.TestCase):
    def test_accepts_curated_raw_page_shape(self):
        self.assertEqual(validate_raw_page(_valid_raw_page()), [])

    def test_rejects_blocked_page_text(self):
        raw = _valid_raw_page()
        raw["title"] = "File Not Found | CK-12 Foundation"
        raw["raw_text"] = "Please wait. Oops, looks like cookies are disabled. No Results Found."

        reasons = validate_raw_page(raw)

        self.assertTrue(any("file not found" in reason for reason in reasons))
        self.assertTrue(any("cookies are disabled" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
