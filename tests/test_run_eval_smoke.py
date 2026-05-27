import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.judge import JudgeResult
from evaluation.scoring import ScoredItem
from evaluation import run_eval


class TestRunEvalSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(__file__).resolve().parent / ".tmp_eval_smoke"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.ds = self.tmp_root / "gold.jsonl"
        self.ds.write_text(
            '{"id":"x1","subject":"math","grade":"M4","question":"Q1","reference_answer":"A1"}\n',
            encoding="utf-8",
        )
        self.results_dir = self.tmp_root / "results"
        self.summaries_dir = self.tmp_root / "summaries"
        self.results_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for p in sorted(self.tmp_root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass

    def test_run_eval_writes_artifacts(self):
        fake_item = ScoredItem(
            item_id="x1",
            subject="math",
            grade="M4",
            question="Q1",
            reference_answer="A1",
            preferred_answer_type="general",
            language="",
            keypoints="",
            model_answer="M1",
            retrieved_context="C1",
            scores=JudgeResult(4, 4, 4, 4, 4, "general", "general", 7, "ok"),
            overall_band=7,
        )

        with patch.object(run_eval, "RESULTS_DIR", self.results_dir), \
             patch.object(run_eval, "SUMMARIES_DIR", self.summaries_dir), \
             patch.object(run_eval, "_run_item", return_value=fake_item):
            summary = run_eval.run_evaluation(dataset_path=self.ds, limit=1)

        self.assertEqual(summary["count"], 1)
        result_files = list(self.results_dir.glob("*.jsonl"))
        summary_files = list(self.summaries_dir.glob("*.json"))
        self.assertEqual(len(result_files), 1)
        self.assertEqual(len(summary_files), 1)

        row = result_files[0].read_text(encoding="utf-8").strip()
        payload = json.loads(row)
        self.assertEqual(payload["id"], "x1")
        self.assertIn("scores", payload)
        self.assertIn("type_alignment", payload["scores"])
        self.assertNotIn("safety", payload["scores"])
        self.assertEqual(payload["overall_band"], 7)

    def test_run_item_passes_generation_metadata(self):
        item = run_eval.EvalItem(
            id="x1",
            subject="physics",
            grade="M4",
            question="Q1",
            reference_answer="A1",
            preferred_answer_type="homework-help",
            language="TH",
            keypoints="show formula; final answer",
        )
        fake_chain = Mock()
        fake_chain.ask.return_value = "M1"
        fake_chain.get_context_docs.return_value = []
        fake_judge = Mock()
        fake_judge.score.return_value = JudgeResult(
            4, 4, 4, 4, 4, "homework-help", "homework-help", 7, "ok"
        )

        with patch.object(run_eval, "get_rag_chain", return_value=fake_chain):
            scored = run_eval._run_item(item, fake_judge)

        fake_chain.ask.assert_called_once_with(
            "Q1",
            subject="physics",
            grade="M4",
            preferred_answer_type="homework-help",
            language="TH",
            keypoints="show formula; final answer",
        )
        self.assertEqual(scored.language, "TH")
        self.assertEqual(scored.keypoints, "show formula; final answer")


if __name__ == "__main__":
    unittest.main()
