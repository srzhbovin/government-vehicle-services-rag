import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_bm25_retrieval import evaluate  # noqa: E402


class EvaluateBM25RetrievalTests(unittest.TestCase):
    def test_evaluate_reports_document_level_recall(self):
        chunks = [
            {
                "chunk_id": "a",
                "document_id": "doc-a",
                "text": "registration renewal insurance proof",
            },
            {
                "chunk_id": "b",
                "document_id": "doc-b",
                "text": "road test appointment schedule",
            },
        ]
        questions = [
            {
                "question": "How do I renew registration with insurance?",
                "gold_document_ids": ["doc-a"],
            }
        ]

        result = evaluate(chunks, questions, cutoffs=(1, 2))

        self.assertEqual(result["questions"], 1)
        self.assertEqual(result["recall@1"], 1.0)
        self.assertEqual(result["recall@2"], 1.0)


if __name__ == "__main__":
    unittest.main()
