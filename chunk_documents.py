#!/usr/bin/env python3
"""Split prepared documents into retrieval chunks for a baseline RAG."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/prepared/dmv_documents.jsonl")
DEFAULT_OUTPUT = Path("data/prepared/dmv_chunks.jsonl")
DEFAULT_MAX_WORDS = 450
DEFAULT_OVERLAP_WORDS = 70
DEFAULT_MIN_TAIL_WORDS = 80

WORD_PATTERN = re.compile(r"\S+")


class ChunkingError(ValueError):
    """Raised when prepared documents cannot be chunked safely."""


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    source: str | None
    text: str
    title: str | None = None
    domain: str | None = None
    url: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceSpanRef:
    span_id: str
    tag: str | None
    title: str | None
    original_start: int | None
    original_end: int | None
    char_start: int
    char_end: int

    def to_record(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "tag": self.tag,
            "title": self.title,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str | None
    domain: str | None
    url: str | None
    source: str | None
    word_start: int
    word_end: int
    char_start: int
    char_end: int
    source_spans: list[SourceSpanRef]

    @property
    def word_count(self) -> int:
        return self.word_end - self.word_start

    @property
    def paragraph_count(self) -> int:
        return len([line for line in self.text.splitlines() if line.strip()])

    def to_record(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "domain": self.domain,
            "url": self.url,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "boundaries": {
                "word_start": self.word_start,
                "word_end": self.word_end,
                "char_start": self.char_start,
                "char_end": self.char_end,
            },
            "stats": {
                "characters": len(self.text),
                "words": self.word_count,
                "paragraphs": self.paragraph_count,
            },
            "source_span_ids": [span.span_id for span in self.source_spans],
            "source_spans": [span.to_record() for span in self.source_spans],
        }


def normalize_inline_text(value: str) -> str:
    """Normalize source span text in the same spirit as document loading."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def read_prepared_documents(path: Path) -> list[PreparedDocument]:
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    if not path.is_file():
        raise ChunkingError(f"expected a JSONL file, got: {path}")

    documents = []
    seen_ids = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise ChunkingError(
            f"{path} is not valid UTF-8 (byte offset {error.start})"
        ) from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ChunkingError(
                f"invalid JSON at {path}:{line_number}: {error.msg}"
            ) from error

        document_id = record.get("document_id")
        text = record.get("text")
        if not document_id:
            raise ChunkingError(f"missing document_id at {path}:{line_number}")
        if not isinstance(text, str) or not text.strip():
            raise ChunkingError(f"missing text at {path}:{line_number}")
        if document_id in seen_ids:
            raise ChunkingError(f"duplicate document_id: {document_id}")
        seen_ids.add(document_id)

        metadata = record.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ChunkingError(f"metadata must be an object at {path}:{line_number}")

        documents.append(
            PreparedDocument(
                document_id=str(document_id),
                source=record.get("source"),
                text=text,
                title=record.get("title"),
                domain=record.get("domain"),
                url=record.get("url"),
                metadata=metadata or {},
            )
        )

    if not documents:
        raise ChunkingError(f"no documents found in: {path}")
    return documents


def iter_words(text: str) -> list[WordSpan]:
    return [
        WordSpan(match.group(0), match.start(), match.end())
        for match in WORD_PATTERN.finditer(text)
    ]


def validate_chunking_options(
    max_words: int,
    overlap_words: int,
    min_tail_words: int,
) -> None:
    if max_words <= 0:
        raise ChunkingError("--max-words must be positive")
    if overlap_words < 0:
        raise ChunkingError("--overlap-words cannot be negative")
    if overlap_words >= max_words:
        raise ChunkingError("--overlap-words must be smaller than --max-words")
    if min_tail_words < 0:
        raise ChunkingError("--min-tail-words cannot be negative")


def sorted_span_items(spans: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        span_id, _ = item
        try:
            return int(span_id), span_id
        except ValueError:
            return sys.maxsize, span_id

    return [
        (span_id, span)
        for span_id, span in sorted(spans.items(), key=sort_key)
        if isinstance(span, dict)
    ]


def find_span_position(
    document_text: str,
    span_text: str,
    original_start: int | None,
    original_end: int | None,
) -> tuple[int, int] | None:
    if not span_text:
        return None

    if original_start is not None:
        window_start = max(0, original_start - 300)
        window_end = len(document_text)
        if original_end is not None:
            window_end = min(len(document_text), original_end + 300)
        local_position = document_text.find(span_text, window_start, window_end)
        if local_position != -1:
            return local_position, local_position + len(span_text)

    global_position = document_text.find(span_text)
    if global_position == -1:
        return None
    return global_position, global_position + len(span_text)


def build_source_span_refs(document: PreparedDocument) -> list[SourceSpanRef]:
    metadata = document.metadata or {}
    raw_spans = metadata.get("spans")
    if not isinstance(raw_spans, dict):
        return []

    refs = []
    for span_id, span in sorted_span_items(raw_spans):
        raw_text = span.get("text_sp")
        if not isinstance(raw_text, str):
            continue

        span_text = normalize_inline_text(raw_text)
        original_start = span.get("start_sp")
        original_end = span.get("end_sp")
        if not isinstance(original_start, int):
            original_start = None
        if not isinstance(original_end, int):
            original_end = None

        position = find_span_position(
            document_text=document.text,
            span_text=span_text,
            original_start=original_start,
            original_end=original_end,
        )
        if position is None:
            continue

        char_start, char_end = position
        refs.append(
            SourceSpanRef(
                span_id=str(span.get("id_sp") or span_id),
                tag=span.get("tag"),
                title=span.get("title"),
                original_start=original_start,
                original_end=original_end,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return refs


def overlapping_source_spans(
    source_spans: list[SourceSpanRef],
    char_start: int,
    char_end: int,
) -> list[SourceSpanRef]:
    return [
        span
        for span in source_spans
        if span.char_start < char_end and span.char_end > char_start
    ]


def chunk_document(
    document: PreparedDocument,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_tail_words: int = DEFAULT_MIN_TAIL_WORDS,
) -> list[Chunk]:
    validate_chunking_options(max_words, overlap_words, min_tail_words)

    words = iter_words(document.text)
    if not words:
        raise ChunkingError(f"document has no words: {document.document_id}")

    source_spans = build_source_span_refs(document)
    chunks = []
    start_word = 0
    chunk_index = 0

    while start_word < len(words):
        end_word = min(start_word + max_words, len(words))
        remaining_words = len(words) - end_word
        if 0 < remaining_words < min_tail_words:
            end_word = len(words)

        char_start = words[start_word].start
        char_end = words[end_word - 1].end
        chunk_text = document.text[char_start:char_end].strip()
        if not chunk_text:
            raise ChunkingError(
                f"empty chunk for {document.document_id} at index {chunk_index}"
            )

        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::chunk-{chunk_index:04d}",
                document_id=document.document_id,
                chunk_index=chunk_index,
                text=chunk_text,
                title=document.title,
                domain=document.domain,
                url=document.url,
                source=document.source,
                word_start=start_word,
                word_end=end_word,
                char_start=char_start,
                char_end=char_end,
                source_spans=overlapping_source_spans(
                    source_spans, char_start, char_end
                ),
            )
        )

        if end_word >= len(words):
            break

        next_start_word = end_word - overlap_words
        if next_start_word <= start_word:
            next_start_word = end_word
        start_word = next_start_word
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: Iterable[PreparedDocument],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_tail_words: int = DEFAULT_MIN_TAIL_WORDS,
) -> list[Chunk]:
    validate_chunking_options(max_words, overlap_words, min_tail_words)

    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                max_words=max_words,
                overlap_words=overlap_words,
                min_tail_words=min_tail_words,
            )
        )
    return chunks


def write_chunks(chunks: Iterable[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(chunk.to_record(), file, ensure_ascii=False)
            file.write("\n")


def print_summary(chunks: list[Chunk], list_limit: int) -> None:
    words = [chunk.word_count for chunk in chunks]
    chunks_without_spans = sum(1 for chunk in chunks if not chunk.source_spans)

    print(f"Chunks:              {len(chunks)}")
    print(f"Words per chunk min: {min(words)}")
    print(f"Words per chunk avg: {statistics.mean(words):.1f}")
    print(f"Words per chunk max: {max(words)}")
    print(f"Chunks w/o spans:    {chunks_without_spans}")
    print()

    for chunk in chunks[:list_limit]:
        label = chunk.title or chunk.document_id
        print(
            f"- {chunk.chunk_id}: {label} — "
            f"{chunk.word_count} words, {len(chunk.source_spans)} spans"
        )
    if len(chunks) > list_limit:
        print(f"... and {len(chunks) - list_limit} more chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split prepared RAG documents into retrieval chunks."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to prepared document JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for chunk JSONL.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help="Maximum target chunk size in words.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help="Number of words repeated between neighboring chunks.",
    )
    parser.add_argument(
        "--min-tail-words",
        type=int,
        default=DEFAULT_MIN_TAIL_WORDS,
        help="Merge the final tail into the previous chunk if it is shorter.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=10,
        help="Maximum number of chunk summaries to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_limit < 0:
        print("Error: --list-limit cannot be negative", file=sys.stderr)
        return 2

    try:
        validate_chunking_options(
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            min_tail_words=args.min_tail_words,
        )
        documents = read_prepared_documents(args.path)
        chunks = chunk_documents(
            documents=documents,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            min_tail_words=args.min_tail_words,
        )
        write_chunks(chunks, args.output)
    except (ChunkingError, FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(chunks, args.list_limit)
    print(f"\nPrepared chunks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
