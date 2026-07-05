"""Build a reproducible standalone DMV RAG evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .settings import PROJECT_ROOT


DEFAULT_SOURCE = (
    PROJECT_ROOT / "data" / "raw" / "multidoc2dial" / "dmv_questions_validation.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "rag_test_set.jsonl"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
CONTENT_STOPWORDS = {
    "about",
    "after",
    "before",
    "from",
    "have",
    "that",
    "their",
    "there",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "your",
}


MANUAL_REGRESSION_CASE = {
    "case_id": "manual_lost_driver_license",
    "question_id": "manual_lost_driver_license",
    "question": "What should I do if I lost my driver license?",
    "reference_answer": (
        "You can replace a lost driver license online, by mail, or at a DMV office. "
        "The regular replacement fee is $17.50. If it was stolen or lost due to a "
        "crime, form MV-78B is required for a free replacement."
    ),
    "gold_document_ids": ["Replace license or permit#1_0"],
    "gold_evidence": [
        "You can replace your license or permit if it was lost, stolen or destroyed.",
        "You can replace online, by mail, or at a DMV office.",
        "The fee is $17.50.",
        "A free replacement after a crime requires police form MV-78B.",
    ],
    "required_terms": ["online", "mail", "office", "17.50", "mv-78b"],
    "source": "manual_regression",
}


def build_test_set(
    source_path: Path = DEFAULT_SOURCE,
    output_path: Path = DEFAULT_OUTPUT,
    size: int = 20,
) -> list[dict[str, Any]]:
    if size < 2:
        raise ValueError("Evaluation set size must be at least 2")
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = []
    for row in rows:
        question_words = str(row.get("question") or "").split()
        answer_words = str(row.get("answer") or "").split()
        answer = " ".join(answer_words)
        evidence = " ".join(
            str(item.get("text") or "") for item in row.get("gold_evidence") or []
        )
        if row.get("history"):
            continue
        if len(row.get("gold_document_ids") or []) != 1:
            continue
        if not row.get("gold_evidence"):
            continue
        if not 5 <= len(question_words) <= 24:
            continue
        if not 3 <= len(answer_words) <= 55:
            continue
        lowered_answer = answer.lower()
        lowered_evidence = evidence.strip().lower()
        if answer.endswith("?") or "here is an example" in lowered_answer:
            continue
        if lowered_evidence.startswith("see "):
            continue
        if lowered_answer.startswith(("if ", "when ")) and len(answer_words) < 12:
            continue
        answer_terms = {
            token
            for token in TOKEN_PATTERN.findall(lowered_answer)
            if len(token) > 3 and token not in CONTENT_STOPWORDS
        }
        evidence_terms = {
            token
            for token in TOKEN_PATTERN.findall(lowered_evidence)
            if len(token) > 3 and token not in CONTENT_STOPWORDS
        }
        if len(evidence_terms) < 4:
            continue
        if answer_terms and len(answer_terms & evidence_terms) / len(answer_terms) < 0.45:
            continue
        candidates.append(row)

    candidates.sort(
        key=lambda row: hashlib.sha256(str(row["question_id"]).encode()).hexdigest()
    )
    selected = []
    used_documents = set()
    for row in candidates:
        document_id = str(row["gold_document_ids"][0])
        if document_id in used_documents:
            continue
        used_documents.add(document_id)
        selected.append(
            {
                "case_id": str(row["question_id"]),
                "question_id": str(row["question_id"]),
                "question": " ".join(str(row["question"]).split()),
                "reference_answer": " ".join(str(row["answer"]).split()),
                "gold_document_ids": [document_id],
                "gold_evidence": [
                    " ".join(str(item["text"]).split())
                    for item in row["gold_evidence"]
                    if item.get("text")
                ],
                "required_terms": [],
                "source": "multidoc2dial_validation",
            }
        )
        if len(selected) == size - 1:
            break

    if len(selected) != size - 1:
        raise ValueError(f"Only {len(selected)} suitable unique-document cases found")
    selected.append(MANUAL_REGRESSION_CASE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args()
    cases = build_test_set(args.source, args.output, args.size)
    print(f"Saved {len(cases)} evaluation cases to {args.output}")


if __name__ == "__main__":
    main()
