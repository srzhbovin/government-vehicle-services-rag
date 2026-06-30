#!/usr/bin/env python3
"""Materialize the best token chunking configuration from the chunking study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from compare_chunking_strategies import ChunkConfig, build_chunks
from evaluate_bm25_retrieval import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS = ROOT / "data" / "current" / "prepared" / "dmv_documents.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "current" / "prepared" / "dmv_chunks_token_120_0.jsonl"


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build token 120/0 chunks.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = read_jsonl(args.documents)
    config = ChunkConfig(
        splitter="token",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        unit="tokens",
    )
    chunks = build_chunks(documents, config)
    write_jsonl(chunks, args.output)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Output:    {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
