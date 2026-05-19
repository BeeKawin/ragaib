import unittest
from pathlib import Path

from evaluation.dataset import load_eval_dataset


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / ".tmp_dataset_tests"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in self.tmp_dir.glob("*"):
            p.unlink(missing_ok=True)

    def test_load_dataset_valid(self):
        path = self.tmp_dir / "gold_valid.jsonl"
        path.write_text(
            '{"id":"a1","subject":"math","grade":"M4","question":"Q","reference_answer":"A"}\n',
            encoding="utf-8",
        )
        items = load_eval_dataset(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "a1")

    def test_load_csv_dataset_valid(self):
        path = self.tmp_dir / "gold_valid.csv"
        path.write_text(
            "id,subject,grade,topic,subtopic,difficulty,question_type,question,reference_answer,expected_keywords,source_topic\n"
            "MATH-001,Mathematics,M4,Sets,Union,Easy,Definition,What is a union?,All elements in either set,union; sets,Sets\n",
            encoding="utf-8",
        )
        items = load_eval_dataset(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "MATH-001")
        self.assertEqual(items[0].subject, "math")
        self.assertIn("expected_keywords=union; sets", items[0].notes)

    def test_load_dataset_invalid_schema(self):
        path = self.tmp_dir / "gold_invalid.jsonl"
        path.write_text('{"id":"a1","question":"Q"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_eval_dataset(path)


if __name__ == "__main__":
    unittest.main()
