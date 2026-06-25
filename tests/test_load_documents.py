import csv
import json
import tempfile
import unittest
from pathlib import Path

from load_documents import (
    DocumentLoadError,
    load_jsonl_documents,
    load_path,
    normalize_text,
    read_text_file,
    write_jsonl,
)


class LoadDocumentsTests(unittest.TestCase):
    def test_normalize_text_removes_extra_whitespace_and_blank_lines(self):
        source = "  First   line \r\n\r\n Second\u00a0line  "
        self.assertEqual(normalize_text(source), "First line\nSecond line")

    def test_loads_valid_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.txt"
            path.write_text("Hello, RAG!\n\nSecond paragraph.", encoding="utf-8")

            documents = load_path(path)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].word_count, 4)
            self.assertEqual(documents[0].paragraph_count, 2)

    def test_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text(" \n\n\t", encoding="utf-8")

            with self.assertRaisesRegex(DocumentLoadError, "empty"):
                read_text_file(path)

    def test_reports_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            load_path(Path("/definitely/missing/source.txt"))

    def test_rejects_non_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.txt"
            path.write_bytes(b"\xff\xfe\xfa")

            with self.assertRaisesRegex(DocumentLoadError, "not valid UTF-8"):
                read_text_file(path)

    def test_maps_ragdoll_files_to_csv_rows_and_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "tablet"
            content_dir = dataset_dir / "content_extract"
            content_dir.mkdir(parents=True)

            with (dataset_dir / "products.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["Product", "Brand", "Model", "URL"])
                writer.writerow(["tablet", "Brand A", "Model A", "https://a.test"])
                writer.writerow(["tablet", "Brand B", "Model B", "https://b.test"])

            (content_dir / "0.txt").write_text("Model A text", encoding="utf-8")
            (content_dir / "1.txt").write_text("Model B text", encoding="utf-8")

            documents = load_path(dataset_dir)
            output_path = Path(temp_dir) / "prepared.jsonl"
            write_jsonl(documents, output_path)

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["metadata"]["brand"] for record in records],
                ["Brand A", "Brand B"],
            )
            self.assertEqual(records[1]["metadata"]["model"], "Model B")
            self.assertEqual(records[0]["stats"]["words"], 3)

    def test_loads_multidoc2dial_jsonl_with_spans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dmv_documents.jsonl"
            records = [
                {
                    "document_id": "dmv-1",
                    "title": "Change your address",
                    "domain": "dmv",
                    "text": "Report your new address within ten days.",
                    "spans": {"1": {"text_sp": "within ten days"}},
                },
                {
                    "document_id": "dmv-2",
                    "title": "Renew a registration",
                    "domain": "dmv",
                    "text": "You can renew online.",
                    "spans": {},
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            documents = load_jsonl_documents(path)

            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].title, "Change your address")
            self.assertEqual(documents[0].domain, "dmv")
            self.assertIn("spans", documents[0].metadata)

    def test_rejects_duplicate_jsonl_document_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "documents.jsonl"
            path.write_text(
                '{"document_id":"same","text":"First"}\n'
                '{"document_id":"same","text":"Second"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DocumentLoadError, "duplicate"):
                load_jsonl_documents(path)


if __name__ == "__main__":
    unittest.main()
