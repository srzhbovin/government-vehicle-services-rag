#!/usr/bin/env python3
"""Load and normalize text documents for the first stage of a baseline RAG."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


class DocumentLoadError(ValueError):
    """Raised when a source document cannot be prepared safely."""


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    text: str
    title: str | None = None
    domain: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def paragraph_count(self) -> int:
        return len(self.text.splitlines())

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["stats"] = {
            "characters": self.char_count,
            "words": self.word_count,
            "paragraphs": self.paragraph_count,
        }
        return record


def normalize_text(raw_text: str) -> str:
    """Normalize Unicode and whitespace while preserving meaningful line breaks."""
    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")

    lines = []
    for line in normalized.splitlines():
        clean_line = " ".join(line.split())
        if clean_line:
            lines.append(clean_line)

    text = "\n".join(lines)
    if not text:
        raise DocumentLoadError("document is empty after normalization")
    return text


def read_text_file(path: Path) -> str:
    """Read one UTF-8 text file and return normalized text."""
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"expected a file, got: {path}")

    try:
        raw_text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DocumentLoadError(
            f"{path} is not valid UTF-8 (byte offset {error.start})"
        ) from error

    return normalize_text(raw_text)


def load_single_document(path: Path) -> Document:
    return Document(
        document_id=path.stem,
        source=str(path),
        text=read_text_file(path),
        title=path.stem,
    )


def load_jsonl_documents(path: Path) -> list[Document]:
    """Load document records from a UTF-8 JSONL file."""
    documents = []
    seen_ids = set()

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise DocumentLoadError(
            f"{path} is not valid UTF-8 (byte offset {error.start})"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DocumentLoadError(
                f"invalid JSON at {path}:{line_number}: {error.msg}"
            ) from error

        document_id = record.get("document_id") or record.get("doc_id")
        raw_text = record.get("text") or record.get("doc_text")
        if not document_id:
            raise DocumentLoadError(
                f"missing document_id at {path}:{line_number}"
            )
        if raw_text is None:
            raise DocumentLoadError(f"missing text at {path}:{line_number}")
        if document_id in seen_ids:
            raise DocumentLoadError(f"duplicate document_id: {document_id}")
        seen_ids.add(document_id)

        core_fields = {
            "document_id",
            "doc_id",
            "text",
            "doc_text",
            "title",
            "domain",
            "url",
        }
        metadata = {
            key: value for key, value in record.items() if key not in core_fields
        }
        documents.append(
            Document(
                document_id=str(document_id),
                source=f"{path.name}:{line_number}",
                text=normalize_text(str(raw_text)),
                title=record.get("title"),
                domain=record.get("domain"),
                url=record.get("url"),
                metadata=metadata,
            )
        )

    if not documents:
        raise DocumentLoadError(f"no documents found in: {path}")
    return documents


def read_product_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"metadata file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Product", "Brand", "Model", "URL"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise DocumentLoadError(f"metadata is missing columns: {missing}")
        return list(reader)


def numeric_txt_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return sys.maxsize, path.name


def load_ragdoll_dataset(dataset_dir: Path) -> list[Document]:
    metadata = read_product_metadata(dataset_dir / "products.csv")
    content_dir = dataset_dir / "content_extract"
    if not content_dir.is_dir():
        raise FileNotFoundError(f"content directory does not exist: {content_dir}")

    source_files = sorted(content_dir.glob("*.txt"), key=numeric_txt_sort_key)
    if not source_files:
        raise DocumentLoadError(f"no .txt documents found in: {content_dir}")

    documents = []
    for path in source_files:
        if not path.stem.isdigit():
            raise DocumentLoadError(
                f"RAGDOLL document name must be a numeric index: {path.name}"
            )

        metadata_index = int(path.stem)
        if metadata_index >= len(metadata):
            raise DocumentLoadError(
                f"{path.name} has no matching row in {dataset_dir / 'products.csv'}"
            )

        row = metadata[metadata_index]
        documents.append(
            Document(
                document_id=f"tablet-{metadata_index}",
                source=str(path.relative_to(dataset_dir)),
                text=read_text_file(path),
                title=f"{row['Brand']} {row['Model']}",
                domain=row["Product"],
                url=row["URL"],
                metadata={"brand": row["Brand"], "model": row["Model"]},
            )
        )

    return documents


def load_path(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() == ".txt":
            return [load_single_document(path)]
        if path.suffix.lower() == ".jsonl":
            return load_jsonl_documents(path)
        raise DocumentLoadError(f"only .txt and .jsonl files are supported: {path}")
    if path.is_dir():
        multidoc_path = path / "dmv_documents.jsonl"
        if multidoc_path.exists():
            return load_jsonl_documents(multidoc_path)
        return load_ragdoll_dataset(path)
    raise DocumentLoadError(f"unsupported path type: {path}")


def write_jsonl(documents: Iterable[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            json.dump(document.to_record(), file, ensure_ascii=False)
            file.write("\n")


def print_summary(
    documents: list[Document], preview_chars: int, list_limit: int
) -> None:
    total_chars = sum(document.char_count for document in documents)
    total_words = sum(document.word_count for document in documents)
    total_paragraphs = sum(document.paragraph_count for document in documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Characters:       {total_chars}")
    print(f"Words:            {total_words}")
    print(f"Paragraphs:       {total_paragraphs}")
    print()

    for document in documents[:list_limit]:
        label = document.title or document.source
        warning = " [very short source]" if document.word_count < 50 else ""
        print(
            f"- {document.document_id}: {label} — "
            f"{document.word_count} words{warning}"
        )
    if len(documents) > list_limit:
        print(f"... and {len(documents) - list_limit} more documents")

    print("\nPreview:")
    preview = documents[0].text[:preview_chars]
    if len(documents[0].text) > preview_chars:
        preview += "..."
    print(preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and normalize documents for a baseline RAG."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/raw/multidoc2dial/dmv_documents.jsonl"),
        help="Path to a .txt file, document JSONL file, or supported dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared/dmv_documents.jsonl"),
        help="Destination for normalized JSONL documents.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of characters shown from the first document.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=10,
        help="Maximum number of document summaries to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preview_chars < 0:
        print("Error: --preview-chars cannot be negative", file=sys.stderr)
        return 2
    if args.list_limit < 0:
        print("Error: --list-limit cannot be negative", file=sys.stderr)
        return 2

    try:
        documents = load_path(args.path)
        write_jsonl(documents, args.output)
    except (DocumentLoadError, FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(documents, args.preview_chars, args.list_limit)
    print(f"\nPrepared corpus: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
