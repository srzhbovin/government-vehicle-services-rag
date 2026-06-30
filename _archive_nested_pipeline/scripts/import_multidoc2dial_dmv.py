#!/usr/bin/env python3
"""Import the DMV domain from the official MultiDoc2Dial archive."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "downloads" / "multidoc2dial.zip"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prepared"
DOMAIN = "dmv"


class ImportErrorDMV(ValueError):
    """Raised when the MultiDoc2Dial archive does not match expectations."""


def normalize_text(raw_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    normalized = "\n".join(line for line in normalized.splitlines() if line)
    if not normalized.strip():
        raise ImportErrorDMV("document text is empty after normalization")
    return normalized.strip()


def read_json_from_zip(archive: Path, member_name: str) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    try:
        with zipfile.ZipFile(archive) as zip_file:
            return json.loads(zip_file.read(member_name))
    except KeyError as error:
        raise ImportErrorDMV(f"archive member is missing: {member_name}") from error


def build_document_records(doc_payload: dict[str, Any]) -> list[dict[str, Any]]:
    domain_docs = doc_payload.get("doc_data", {}).get(DOMAIN)
    if not isinstance(domain_docs, dict):
        raise ImportErrorDMV("doc_data.dmv is missing or invalid")

    documents = []
    for _, source in sorted(domain_docs.items(), key=lambda item: item[0]):
        document_id = source.get("doc_id")
        text = source.get("doc_text")
        if not document_id or not isinstance(text, str):
            raise ImportErrorDMV("DMV document is missing doc_id or doc_text")

        documents.append(
            {
                "document_id": document_id,
                "title": source.get("title"),
                "domain": source.get("domain", DOMAIN),
                "url": None,
                "source": "MultiDoc2Dial DMV",
                "text": normalize_text(text),
                "metadata": {
                    "spans": source.get("spans", {}),
                },
            }
        )
    return documents


def collect_gold_evidence(
    references: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = []
    seen = set()
    for reference in references:
        doc_id = reference.get("doc_id")
        span_id = str(reference.get("id_sp"))
        if not doc_id or not span_id:
            continue

        key = (doc_id, span_id)
        if key in seen:
            continue
        seen.add(key)

        source_document = documents_by_id.get(doc_id)
        source_span = (
            source_document.get("metadata", {}).get("spans", {}).get(span_id)
            if source_document
            else None
        )
        evidence.append(
            {
                "document_id": doc_id,
                "span_id": span_id,
                "label": reference.get("label"),
                "text": source_span.get("text_sp") if isinstance(source_span, dict) else None,
                "title": source_span.get("title") if isinstance(source_span, dict) else None,
            }
        )
    return evidence


def build_question_records(
    dial_payload: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    dialogs = dial_payload.get("dial_data", {}).get(DOMAIN)
    if not isinstance(dialogs, list):
        raise ImportErrorDMV("dial_data.dmv is missing or invalid")

    records = []
    for dialog in dialogs:
        turns = dialog.get("turns", [])
        if not isinstance(turns, list):
            continue

        for turn_index, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue

            if turn_index + 1 >= len(turns):
                continue
            answer_turn = turns[turn_index + 1]
            if answer_turn.get("role") != "agent":
                continue

            answer_references = answer_turn.get("references", [])
            if not isinstance(answer_references, list):
                answer_references = []

            gold_document_ids = sorted(
                {
                    reference.get("doc_id")
                    for reference in answer_references
                    if reference.get("doc_id")
                }
            )
            records.append(
                {
                    "question_id": f"{dialog.get('dial_id')}:{turn.get('turn_id')}",
                    "dial_id": dialog.get("dial_id"),
                    "turn_id": turn.get("turn_id"),
                    "question": turn.get("utterance", ""),
                    "history": [
                        {
                            "role": previous.get("role"),
                            "utterance": previous.get("utterance", ""),
                        }
                        for previous in turns[:turn_index]
                    ],
                    "answer": answer_turn.get("utterance", ""),
                    "gold_document_ids": gold_document_ids,
                    "gold_evidence": collect_gold_evidence(
                        answer_references, documents_by_id
                    ),
                }
            )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MultiDoc2Dial DMV data.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to multidoc2dial.zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prepared DMV JSONL files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc_payload = read_json_from_zip(args.archive, "multidoc2dial/multidoc2dial_doc.json")
    documents = build_document_records(doc_payload)
    documents_by_id = {document["document_id"]: document for document in documents}

    write_jsonl(documents, args.output_dir / "dmv_documents.jsonl")

    question_counts = {}
    for split in ("train", "validation", "test"):
        dial_payload = read_json_from_zip(
            args.archive, f"multidoc2dial/multidoc2dial_dial_{split}.json"
        )
        questions = build_question_records(dial_payload, documents_by_id)
        write_jsonl(questions, args.output_dir / f"dmv_questions_{split}.jsonl")
        question_counts[split] = len(questions)

    manifest = {
        "dataset": "MultiDoc2Dial",
        "domain": DOMAIN,
        "documents": len(documents),
        "questions": question_counts,
        "source_archive": str(args.archive),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_words = sum(len(document["text"].split()) for document in documents)
    print(f"Documents: {len(documents)}")
    print(f"Words:     {total_words}")
    for split, count in question_counts.items():
        print(f"{split}: {count} QA pairs")
    print(f"Output:    {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
