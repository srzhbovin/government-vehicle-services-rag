#!/usr/bin/env python3
"""Recursive character chunking for RAG documents.

Defaults intentionally mirror the requested company-style settings:

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "prepared" / "dmv_documents.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "prepared" / "dmv_chunks_recursive.jsonl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""]
WORD_PATTERN = re.compile(r"\S+")


class ChunkingError(ValueError):
    """Raised when a document cannot be chunked safely."""


@dataclass(frozen=True)
class TextPart:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class WordSpan:
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
            },
            "source_span_ids": [span.span_id for span in self.source_spans],
            "source_spans": [span.to_record() for span in self.source_spans],
        }


def normalize_inline_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def validate_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ChunkingError("--chunk-size must be positive")
    if chunk_overlap < 0:
        raise ChunkingError("--chunk-overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("--chunk-overlap must be smaller than --chunk-size")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ChunkingError(
                    f"invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
    if not records:
        raise ChunkingError(f"no records found in: {path}")
    return records


def split_with_separator(
    text: str,
    offset: int,
    separator: str,
    remaining_separators: list[str],
    chunk_size: int,
) -> list[TextPart]:
    parts = []
    position = 0
    while position < len(text):
        found = text.find(separator, position)
        end = len(text) if found == -1 else found + len(separator)
        if end > position:
            piece = text[position:end]
            piece_start = offset + position
            if len(piece) > chunk_size and remaining_separators:
                parts.extend(
                    recursive_atomic_parts(
                        piece,
                        piece_start,
                        chunk_size,
                        remaining_separators,
                    )
                )
            else:
                parts.append(TextPart(piece_start, offset + end))
        if found == -1:
            break
        position = end
    return parts


def recursive_atomic_parts(
    text: str,
    offset: int,
    chunk_size: int,
    separators: list[str] | None = None,
) -> list[TextPart]:
    """Split text recursively into mergeable atomic parts."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [TextPart(offset, offset + len(text))]

    separators = separators or SEPARATORS
    separator = next((candidate for candidate in separators if candidate and candidate in text), "")
    if separator == "":
        return [
            TextPart(offset + start, offset + min(start + chunk_size, len(text)))
            for start in range(0, len(text), chunk_size)
        ]

    next_separators = separators[separators.index(separator) + 1 :]
    return split_with_separator(text, offset, separator, next_separators, chunk_size)


def trailing_overlap_parts(parts: list[TextPart], chunk_overlap: int) -> list[TextPart]:
    if chunk_overlap <= 0:
        return []

    selected = []
    total = 0
    for part in reversed(parts):
        if selected and total + part.length > chunk_overlap:
            break
        selected.insert(0, part)
        total += part.length
        if total >= chunk_overlap:
            break
    return selected


def merge_parts(parts: list[TextPart], chunk_size: int, chunk_overlap: int) -> list[TextPart]:
    chunks = []
    current: list[TextPart] = []

    for part in parts:
        while current and sum(piece.length for piece in current) + part.length > chunk_size:
            chunks.append(TextPart(current[0].start, current[-1].end))
            current = trailing_overlap_parts(current, chunk_overlap)
            while current and sum(piece.length for piece in current) + part.length > chunk_size:
                current.pop(0)

        current.append(part)

    if current:
        chunks.append(TextPart(current[0].start, current[-1].end))
    return chunks


def trim_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(match.start(), match.end()) for match in WORD_PATTERN.finditer(text)]


def word_boundaries(words: list[WordSpan], char_start: int, char_end: int) -> tuple[int, int]:
    word_ends = [word.end for word in words]
    word_starts = [word.start for word in words]
    return (
        bisect.bisect_right(word_ends, char_start),
        bisect.bisect_left(word_starts, char_end),
    )


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


def build_source_span_refs(document: dict[str, Any]) -> list[SourceSpanRef]:
    text = document["text"]
    raw_spans = document.get("metadata", {}).get("spans", {})
    if not isinstance(raw_spans, dict):
        return []

    refs = []
    for span_id, span in raw_spans.items():
        if not isinstance(span, dict) or not isinstance(span.get("text_sp"), str):
            continue

        original_start = span.get("start_sp")
        original_end = span.get("end_sp")
        if not isinstance(original_start, int):
            original_start = None
        if not isinstance(original_end, int):
            original_end = None

        span_text = normalize_inline_text(span["text_sp"])
        position = find_span_position(text, span_text, original_start, original_end)
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
    document: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    validate_options(chunk_size, chunk_overlap)
    text = document.get("text")
    document_id = document.get("document_id")
    if not document_id:
        raise ChunkingError("document is missing document_id")
    if not isinstance(text, str) or not text.strip():
        raise ChunkingError(f"document has empty text: {document_id}")

    atomic_parts = recursive_atomic_parts(text, 0, chunk_size)
    rough_chunks = merge_parts(atomic_parts, chunk_size, chunk_overlap)
    words = word_spans(text)
    source_spans = build_source_span_refs(document)

    chunks = []
    seen_boundaries = set()
    for rough_chunk in rough_chunks:
        char_start, char_end = trim_boundaries(text, rough_chunk.start, rough_chunk.end)
        if char_start >= char_end or (char_start, char_end) in seen_boundaries:
            continue
        seen_boundaries.add((char_start, char_end))

        word_start, word_end = word_boundaries(words, char_start, char_end)
        chunk_text = text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}::chunk-{len(chunks):04d}",
                document_id=str(document_id),
                chunk_index=len(chunks),
                text=chunk_text,
                title=document.get("title"),
                domain=document.get("domain"),
                url=document.get("url"),
                source=document.get("source"),
                word_start=word_start,
                word_end=word_end,
                char_start=char_start,
                char_end=char_end,
                source_spans=overlapping_source_spans(
                    source_spans, char_start, char_end
                ),
            )
        )

    if not chunks:
        raise ChunkingError(f"no chunks produced for document: {document_id}")
    return chunks


def chunk_documents(
    documents: Iterable[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    validate_options(chunk_size, chunk_overlap)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks


def write_jsonl(chunks: Iterable[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            json.dump(chunk.to_record(), file, ensure_ascii=False)
            file.write("\n")


def print_summary(chunks: list[Chunk]) -> None:
    char_counts = [len(chunk.text) for chunk in chunks]
    word_counts = [chunk.word_count for chunk in chunks]
    print(f"Chunks:              {len(chunks)}")
    print(f"Characters min/avg/max: {min(char_counts)} / {statistics.mean(char_counts):.1f} / {max(char_counts)}")
    print(f"Words min/avg/max:      {min(word_counts)} / {statistics.mean(word_counts):.1f} / {max(word_counts)}")
    print(f"Chunks w/o spans:    {sum(1 for chunk in chunks if not chunk.source_spans)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive chunking for RAG documents.")
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
        help="Destination for recursive chunk JSONL.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Target character overlap between neighboring chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = read_jsonl(args.path)
        chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
        write_jsonl(chunks, args.output)
    except (ChunkingError, FileNotFoundError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_summary(chunks)
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
