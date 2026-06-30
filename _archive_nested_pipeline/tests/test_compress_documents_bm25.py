import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compress_documents_bm25 import (  # noqa: E402
    CompressionError,
    compress_documents,
    split_segments,
    tokenize,
)


class CompressDocumentsBM25Tests(unittest.TestCase):
    def test_tokenize_removes_stopwords(self):
        self.assertEqual(tokenize("The DMV registration renewal form"), ["dmv", "registration", "renewal", "form"])

    def test_split_segments_falls_back_to_sentence_like_units(self):
        segments = split_segments("First sentence. Second sentence! Third sentence?")

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].text, "First sentence.")

    def test_compress_documents_keeps_important_sentence_and_metadata(self):
        documents = [
            {
                "document_id": "dmv-1",
                "text": (
                    "Generic introduction for the page. "
                    "Vehicle registration renewal requires insurance proof and a valid plate. "
                    "Office hours are listed on the website. "
                    "Registration renewal can be completed online before expiration."
                ),
                "metadata": {
                    "spans": {
                        "1": {
                            "id_sp": "1",
                            "text_sp": "Vehicle registration renewal requires insurance proof and a valid plate.",
                            "title": "Registration",
                        },
                        "2": {
                            "id_sp": "2",
                            "text_sp": "This span is not present in selected text.",
                            "title": "Missing",
                        },
                    }
                },
            },
            {
                "document_id": "dmv-2",
                "text": "Driver license appointments can be changed online. Road tests have separate rules.",
                "metadata": {"spans": {}},
            },
        ]

        compact = compress_documents(
            documents,
            target_ratio=0.35,
            top_terms=10,
            min_segments=2,
            max_segments=2,
            keep_first_segments=0,
        )

        self.assertEqual(len(compact), 2)
        self.assertIn("registration renewal", compact[0]["text"])
        self.assertLess(len(compact[0]["text"]), len(documents[0]["text"]))
        self.assertIn("compression", compact[0]["metadata"])
        self.assertIn("1", compact[0]["metadata"]["spans"])
        self.assertNotIn("2", compact[0]["metadata"]["spans"])

    def test_rejects_invalid_ratio(self):
        with self.assertRaisesRegex(CompressionError, "target-ratio"):
            compress_documents(
                [{"document_id": "x", "text": "One. Two.", "metadata": {}}],
                target_ratio=0,
            )


if __name__ == "__main__":
    unittest.main()
