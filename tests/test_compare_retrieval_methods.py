import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "src" / "rag_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from compare_retrieval_methods import (  # noqa: E402
    BM25Retriever,
    FaissFlatRetriever,
    RankedItem,
    TfidfEncoder,
    cosine_search,
    evaluate_rankings,
    lexical_tokenize,
    reciprocal_rank_fusion,
)


class RetrievalMethodsTests(unittest.TestCase):
    def test_tokenizer_normalizes_case_and_removes_stopwords(self):
        self.assertEqual(
            lexical_tokenize("How CAN I renew my driver's license?"),
            ["renew", "driver's", "license"],
        )

    def test_bm25_prefers_document_with_query_terms(self):
        retriever = BM25Retriever(
            [
                "Renew a vehicle registration online.",
                "Replace a lost driver license.",
                "Pay a traffic ticket.",
            ]
        )

        result = retriever.search("How do I replace my lost license?", top_k=2)

        self.assertEqual(result[0].chunk_index, 1)
        self.assertGreater(result[0].score, result[1].score)

    def test_tfidf_encoder_returns_normalized_vectors(self):
        encoder = TfidfEncoder(max_features=20)
        encoder.fit(["vehicle registration renewal", "driver license replacement"])

        vectors = encoder.encode_documents(
            ["vehicle registration renewal", "driver license replacement"]
        )

        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])

    def test_exact_cosine_and_faiss_agree_without_ties(self):
        documents = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
        documents /= np.linalg.norm(documents, axis=1, keepdims=True)
        queries = np.asarray([[1.0, 0.0]], dtype=np.float32)

        cosine = cosine_search(documents, queries, top_k=3)[0]
        faiss = FaissFlatRetriever(documents).search(queries, top_k=3)[0]

        self.assertEqual(
            [item.chunk_index for item in cosine],
            [item.chunk_index for item in faiss],
        )

    def test_rrf_combines_rankings_and_respects_weights(self):
        first = [RankedItem(0, 10.0), RankedItem(1, 9.0)]
        second = [RankedItem(1, 0.9), RankedItem(2, 0.8)]

        equal = reciprocal_rank_fusion([first, second], top_k=3, rrf_k=60)
        first_only = reciprocal_rank_fusion(
            [first, second], top_k=3, rrf_k=60, weights=[1.0, 0.0]
        )

        self.assertEqual(equal[0].chunk_index, 1)
        self.assertEqual(first_only[0].chunk_index, 0)

    def test_metrics_use_gold_document_ids(self):
        chunks = [
            {"chunk_id": "a-0", "document_id": "a", "text": "A"},
            {"chunk_id": "b-0", "document_id": "b", "text": "B"},
        ]
        questions = [
            {
                "question_id": "q1",
                "question": "question",
                "gold_document_ids": ["b"],
            }
        ]
        rankings = [[RankedItem(0, 1.0), RankedItem(1, 0.5)]]

        metrics, details = evaluate_rankings(chunks, questions, rankings)

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr@10"], 0.5)
        self.assertEqual(details[0]["first_hit_rank"], 2)


if __name__ == "__main__":
    unittest.main()
