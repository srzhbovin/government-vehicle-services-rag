import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.evaluation import (  # noqa: E402
    citation_numbers,
    citations_are_valid,
    error_categories,
    exact_match,
    reciprocal_rank,
    retrieval_hit,
    retrieval_recall,
    token_scores,
)


class EvaluationTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_punctuation(self):
        self.assertEqual(exact_match("Renew online!", "renew online"), 1.0)

    def test_token_scores_return_precision_recall_and_f1(self):
        precision, recall, f1 = token_scores("renew online now", "renew online")

        self.assertAlmostEqual(precision, 2 / 3)
        self.assertEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.8)

    def test_retrieval_metrics_use_gold_documents(self):
        retrieved = ["wrong", "gold", "other"]
        gold = ["gold"]

        self.assertFalse(retrieval_hit(retrieved, gold, 1))
        self.assertTrue(retrieval_hit(retrieved, gold, 3))
        self.assertEqual(retrieval_recall(retrieved, gold, 3), 1.0)
        self.assertEqual(reciprocal_rank(retrieved, gold), 0.5)

    def test_citations_are_collected_from_answer_and_source(self):
        citations = citation_numbers("Use the online service [2].", "[1], [2]")

        self.assertEqual(citations, [1, 2])
        self.assertTrue(citations_are_valid(citations, 2))
        self.assertFalse(citations_are_valid([3], 2))

    def test_error_categories_separate_retrieval_and_answer_failures(self):
        categories = error_categories(
            {
                "generation_error": None,
                "retrieval_hit_at_5": False,
                "parse_success": True,
                "schema_valid": True,
                "citation_valid": False,
                "answer": "Unrelated response",
                "semantic_similarity": 0.2,
                "reference_token_recall": 0.1,
                "confidence": 0.9,
            }
        )

        self.assertIn("retrieval_miss", categories)
        self.assertIn("citation_error", categories)
        self.assertIn("low_reference_coverage", categories)


if __name__ == "__main__":
    unittest.main()
