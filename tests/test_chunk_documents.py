import json
import tempfile
import unittest
from pathlib import Path

from chunk_documents import (
    ChunkingError,
    PreparedDocument,
    chunk_document,
    chunk_documents,
    read_prepared_documents,
    write_chunks,
)


class ChunkDocumentsTests(unittest.TestCase):
    def test_short_document_becomes_one_chunk_with_metadata_and_source_span(self):
        document = PreparedDocument(
            document_id="dmv-short",
            source="dmv_documents.jsonl:1",
            title="Short DMV page",
            domain="dmv",
            url="https://dmv.test/page",
            text="Alpha beta gamma delta epsilon.",
            metadata={
                "spans": {
                    "1": {
                        "id_sp": "s1",
                        "tag": "p",
                        "title": "Small section",
                        "start_sp": 6,
                        "end_sp": 16,
                        "text_sp": "beta gamma",
                    }
                }
            },
        )

        chunks = chunk_document(document, max_words=10, overlap_words=2)
        record = chunks[0].to_record()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(record["chunk_id"], "dmv-short::chunk-0000")
        self.assertEqual(record["document_id"], "dmv-short")
        self.assertEqual(record["title"], "Short DMV page")
        self.assertEqual(record["domain"], "dmv")
        self.assertEqual(record["url"], "https://dmv.test/page")
        self.assertEqual(record["stats"]["words"], 5)
        self.assertEqual(record["source_span_ids"], ["s1"])
        self.assertEqual(record["source_spans"][0]["title"], "Small section")

    def test_splits_document_with_word_overlap_and_boundaries(self):
        document = PreparedDocument(
            document_id="dmv-long",
            source=None,
            title=None,
            domain="dmv",
            url=None,
            text=" ".join(f"w{i}" for i in range(12)),
            metadata={},
        )

        chunks = chunk_document(
            document,
            max_words=5,
            overlap_words=2,
            min_tail_words=0,
        )

        self.assertEqual([chunk.word_start for chunk in chunks], [0, 3, 6, 9])
        self.assertEqual([chunk.word_end for chunk in chunks], [5, 8, 11, 12])
        self.assertEqual(chunks[0].text, "w0 w1 w2 w3 w4")
        self.assertEqual(chunks[1].text, "w3 w4 w5 w6 w7")
        self.assertEqual(chunks[-1].text, "w9 w10 w11")

    def test_merges_tiny_final_tail_into_previous_chunk(self):
        document = PreparedDocument(
            document_id="dmv-tail",
            source=None,
            title=None,
            domain="dmv",
            url=None,
            text=" ".join(f"w{i}" for i in range(11)),
            metadata={},
        )

        chunks = chunk_document(
            document,
            max_words=5,
            overlap_words=1,
            min_tail_words=3,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.word_count for chunk in chunks], [5, 7])
        self.assertEqual(chunks[1].text, "w4 w5 w6 w7 w8 w9 w10")

    def test_rejects_overlap_that_is_not_smaller_than_chunk_size(self):
        document = PreparedDocument(
            document_id="dmv-bad",
            source=None,
            title=None,
            domain="dmv",
            url=None,
            text="one two three",
            metadata={},
        )

        with self.assertRaisesRegex(ChunkingError, "overlap"):
            chunk_document(document, max_words=5, overlap_words=5)

    def test_reads_prepared_jsonl_and_writes_chunk_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "documents.jsonl"
            output_path = Path(temp_dir) / "chunks.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "document_id": "dmv-jsonl",
                        "source": "source.jsonl:1",
                        "title": "JSONL page",
                        "domain": "dmv",
                        "url": None,
                        "text": "one two three four five six",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            documents = read_prepared_documents(input_path)
            chunks = chunk_documents(
                documents,
                max_words=4,
                overlap_words=1,
                min_tail_words=0,
            )
            write_chunks(chunks, output_path)

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["document_id"], "dmv-jsonl")
            self.assertEqual(records[0]["boundaries"]["word_start"], 0)
            self.assertEqual(records[1]["boundaries"]["word_start"], 3)

    def test_rejects_duplicate_prepared_document_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "documents.jsonl"
            input_path.write_text(
                '{"document_id":"same","text":"First text"}\n'
                '{"document_id":"same","text":"Second text"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ChunkingError, "duplicate"):
                read_prepared_documents(input_path)


if __name__ == "__main__":
    unittest.main()
