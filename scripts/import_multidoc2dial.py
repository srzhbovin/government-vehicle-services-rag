#!/usr/bin/env python3
"""Extract one MultiDoc2Dial domain into compact, RAG-friendly JSONL files."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


DATASET_URL = (
    "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
)
VALID_DOMAINS = ("dmv", "ssa", "va", "studentaid")
SPLITS = ("train", "validation", "test")


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
            count += 1
    return count


def ensure_archive(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, path)


def read_json_from_zip(
    archive: zipfile.ZipFile, member: str
) -> dict[str, Any]:
    try:
        with archive.open(member) as file:
            return json.load(file)
    except KeyError as error:
        raise ValueError(f"archive does not contain {member}") from error


def build_document_records(
    documents: dict[str, dict[str, Any]], domain: str
) -> Iterable[dict[str, Any]]:
    for document_id in sorted(documents):
        document = documents[document_id]
        yield {
            "document_id": document["doc_id"],
            "title": document["title"],
            "domain": domain,
            "text": document["doc_text"],
            "spans": document["spans"],
        }


def get_reference_evidence(
    references: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = []
    for reference in references:
        document_id = reference["doc_id"]
        span_id = reference["id_sp"]
        span = documents.get(document_id, {}).get("spans", {}).get(span_id, {})
        evidence.append(
            {
                "document_id": document_id,
                "span_id": span_id,
                "label": reference.get("label"),
                "text": span.get("text_sp"),
                "section_title": span.get("title"),
            }
        )
    return evidence


def build_question_records(
    dialogues: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    domain: str,
    split: str,
) -> Iterable[dict[str, Any]]:
    for dialogue in dialogues:
        turns = dialogue["turns"]
        for index, turn in enumerate(turns):
            if turn["role"] != "user":
                continue

            answer_turn = (
                turns[index + 1]
                if index + 1 < len(turns) and turns[index + 1]["role"] == "agent"
                else None
            )
            if answer_turn is None:
                continue

            references = answer_turn.get("references", [])
            evidence = get_reference_evidence(references, documents)
            gold_document_ids = list(
                dict.fromkeys(item["document_id"] for item in evidence)
            )

            yield {
                "question_id": f"{dialogue['dial_id']}:{turn['turn_id']}",
                "dialogue_id": dialogue["dial_id"],
                "turn_id": turn["turn_id"],
                "domain": domain,
                "split": split,
                "question": turn["utterance"],
                "history": [
                    {
                        "role": previous_turn["role"],
                        "text": previous_turn["utterance"],
                    }
                    for previous_turn in turns[:index]
                ],
                "answer": answer_turn["utterance"],
                "gold_document_ids": gold_document_ids,
                "gold_evidence": evidence,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local MultiDoc2Dial domain for baseline RAG."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/downloads/multidoc2dial.zip"),
        help="Path to the official MultiDoc2Dial ZIP archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/multidoc2dial"),
        help="Directory for extracted JSONL files.",
    )
    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        default="dmv",
        help="Domain to extract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_archive(args.archive)

    with zipfile.ZipFile(args.archive) as archive:
        document_data = read_json_from_zip(
            archive, "multidoc2dial/multidoc2dial_doc.json"
        )["doc_data"]
        documents = document_data[args.domain]

        document_count = write_jsonl(
            build_document_records(documents, args.domain),
            args.output_dir / f"{args.domain}_documents.jsonl",
        )
        print(f"Documents: {document_count}")

        split_counts = {}
        for split in SPLITS:
            dialogue_data = read_json_from_zip(
                archive,
                f"multidoc2dial/multidoc2dial_dial_{split}.json",
            )["dial_data"][args.domain]
            question_count = write_jsonl(
                build_question_records(
                    dialogue_data, documents, args.domain, split
                ),
                args.output_dir / f"{args.domain}_questions_{split}.jsonl",
            )
            split_counts[split] = question_count
            print(f"{split.title()} questions: {question_count}")

    manifest = {
        "dataset": "MultiDoc2Dial",
        "version": "1.0",
        "domain": args.domain,
        "source": DATASET_URL,
        "documents": document_count,
        "questions": split_counts,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
