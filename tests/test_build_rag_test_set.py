import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.build_rag_test_set import build_test_set  # noqa: E402


class BuildRAGTestSetTests(unittest.TestCase):
    def test_builds_standalone_set_and_adds_regression_case(self):
        row = {
            "question_id": "q-1",
            "question": "Do I need insurance to register my vehicle?",
            "history": [],
            "answer": "The DMV requires liability insurance to register a vehicle.",
            "gold_document_ids": ["insurance-doc"],
            "gold_evidence": [
                {"text": "The DMV requires liability insurance to register a vehicle."}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "test_set.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            cases = build_test_set(source, output, size=2)

            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["gold_document_ids"], ["insurance-doc"])
            self.assertEqual(cases[1]["case_id"], "manual_lost_driver_license")
            self.assertIn("mv-78b", cases[1]["required_terms"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
