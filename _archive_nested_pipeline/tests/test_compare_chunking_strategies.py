import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_chunking_strategies import (  # noqa: E402
    ChunkConfig,
    build_chunks,
    character_boundaries,
    run_experiment,
    semantic_boundaries,
    token_boundaries,
)


class CompareChunkingStrategiesTests(unittest.TestCase):
    def test_character_boundaries_use_character_overlap(self):
        boundaries = character_boundaries("abcdefghij", chunk_size=5, chunk_overlap=2)

        self.assertEqual(
            [(boundary.start, boundary.end) for boundary in boundaries],
            [(0, 5), (3, 8), (6, 10)],
        )

    def test_token_boundaries_use_token_overlap(self):
        boundaries = token_boundaries(
            "one two three four five six",
            chunk_size=3,
            chunk_overlap=1,
        )

        self.assertEqual(len(boundaries), 3)
        self.assertEqual(boundaries[0].start, 0)
        self.assertEqual(boundaries[1].start, len("one two "))

    def test_semantic_boundaries_produce_non_empty_chunks(self):
        text = (
            "Registration renewal requires insurance proof. "
            "You can renew registration online. "
            "Road test appointments use a different system. "
            "You can reschedule road tests online."
        )

        boundaries = semantic_boundaries(
            text,
            chunk_size=90,
            chunk_overlap=20,
            threshold=0.05,
        )

        self.assertGreaterEqual(len(boundaries), 2)
        self.assertTrue(all(boundary.start < boundary.end for boundary in boundaries))

    def test_build_chunks_preserves_document_id_and_strategy_metadata(self):
        document = {
            "document_id": "doc-1",
            "title": "Doc",
            "domain": "dmv",
            "text": "one two three four five six",
            "metadata": {},
        }
        config = ChunkConfig("token", chunk_size=3, chunk_overlap=1, unit="tokens")

        chunks = build_chunks([document], config)

        self.assertEqual(chunks[0]["document_id"], "doc-1")
        self.assertEqual(chunks[0]["splitter"], "token")
        self.assertEqual(chunks[0]["chunk_size"], 3)

    def test_run_experiment_returns_retrieval_metrics(self):
        documents = [
            {
                "document_id": "registration",
                "text": "registration renewal insurance proof",
                "metadata": {},
            },
            {
                "document_id": "road-test",
                "text": "road test appointment schedule",
                "metadata": {},
            },
        ]
        questions = [
            {
                "question": "How do I renew registration?",
                "gold_document_ids": ["registration"],
            }
        ]
        config = ChunkConfig("character", chunk_size=80, chunk_overlap=10, unit="characters")

        result = run_experiment(documents, questions, config)

        self.assertEqual(result["questions"], 1)
        self.assertIn("recall@1", result)
        self.assertEqual(result["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
