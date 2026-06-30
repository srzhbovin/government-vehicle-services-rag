import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recursive_chunk_documents import (  # noqa: E402
    ChunkingError,
    chunk_document,
    chunk_documents,
    read_jsonl,
    write_jsonl,
)


class RecursiveChunkDocumentsTests(unittest.TestCase):
    def test_recursive_chunking_keeps_chunks_under_character_limit(self):
        document = {
            "document_id": "doc-1",
            "title": "Registration",
            "domain": "dmv",
            "url": "https://example.test",
            "source": "test",
            "text": (
                "First sentence about registration renewal. "
                "Second sentence about insurance verification. "
                "Third sentence about license plates and fees. "
                "Fourth sentence about online service appointments."
            ),
            "metadata": {},
        }

        chunks = chunk_document(document, chunk_size=90, chunk_overlap=30)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 90 for chunk in chunks))
        self.assertEqual(chunks[0].document_id, "doc-1")
        self.assertEqual(chunks[0].domain, "dmv")

    def test_neighboring_chunks_have_overlap(self):
        document = {
            "document_id": "doc-overlap",
            "text": " ".join(f"word{i}" for i in range(80)),
            "metadata": {},
        }

        chunks = chunk_document(document, chunk_size=120, chunk_overlap=50)

        self.assertGreater(len(chunks), 2)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(current.char_start, previous.char_end)

    def test_source_spans_are_attached_to_overlapping_chunks(self):
        document = {
            "document_id": "doc-span",
            "text": "Alpha beta gamma. Renew registration online before expiration.",
            "metadata": {
                "spans": {
                    "7": {
                        "id_sp": "7",
                        "tag": "p",
                        "title": "Renewal",
                        "start_sp": 18,
                        "end_sp": 62,
                        "text_sp": "Renew registration online before expiration.",
                    }
                }
            },
        }

        chunks = chunk_document(document, chunk_size=80, chunk_overlap=20)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_spans[0].span_id, "7")
        self.assertEqual(chunks[0].to_record()["source_span_ids"], ["7"])

    def test_rejects_invalid_overlap(self):
        document = {"document_id": "bad", "text": "one two three", "metadata": {}}

        with self.assertRaisesRegex(ChunkingError, "overlap"):
            chunk_document(document, chunk_size=100, chunk_overlap=100)

    def test_reads_and_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "documents.jsonl"
            output_path = Path(temp_dir) / "chunks.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "document_id": "jsonl-doc",
                        "text": "One sentence. Two sentence. Three sentence.",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            documents = read_jsonl(input_path)
            chunks = chunk_documents(documents, chunk_size=40, chunk_overlap=10)
            write_jsonl(chunks, output_path)
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(documents), 1)
            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[0]["document_id"], "jsonl-doc")


if __name__ == "__main__":
    unittest.main()
