"""Build runtime embeddings and a FAISS Flat index for the selected chunks."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from .compare_retrieval_methods import read_jsonl
from .retriever import chunk_search_text, file_sha256
from .settings import Settings


def build_index(settings: Settings) -> dict[str, object]:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies before building the index"
        ) from error

    chunks = read_jsonl(settings.chunks_path)
    texts = [chunk_search_text(chunk) for chunk in chunks]
    model_kwargs = {"device": settings.device} if settings.device else {}
    model = SentenceTransformer(settings.embedding_model, **model_kwargs)

    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    build_ms = (time.perf_counter() - started) * 1000

    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(settings.embeddings_path, vectors, allow_pickle=False)
    faiss.write_index(index, str(settings.index_path))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_path": str(settings.chunks_path.relative_to(settings.project_root)),
        "chunks_sha256": file_sha256(settings.chunks_path),
        "chunks": len(chunks),
        "embedding_backend": "sentence-transformers",
        "embedding_model": settings.embedding_model,
        "embedding_text": "title_plus_text",
        "dimension": int(vectors.shape[1]),
        "faiss_index": str(settings.index_path.relative_to(settings.project_root)),
        "faiss_index_size_bytes": settings.index_path.stat().st_size,
        "runtime_retrieval_method": (
            "hybrid_reranked" if settings.use_reranker else "hybrid_rrf"
        ),
        "bm25": {"k1": 1.5, "b": 0.75},
        "hybrid": {
            "rrf_k": settings.rrf_k,
            "bm25_weight": settings.bm25_weight,
        },
        "reranker_model": settings.reranker_model if settings.use_reranker else None,
        "reranker_candidate_k": (
            settings.reranker_candidate_k if settings.use_reranker else None
        ),
        "build_ms": build_ms,
    }
    settings.index_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the runtime RAG FAISS index")
    parser.add_argument("--env-file", help="Optional path to an environment file")
    args = parser.parse_args()
    settings = Settings.from_env(args.env_file)
    metadata = build_index(settings)
    print(
        f"Built {metadata['chunks']} vectors, dimension={metadata['dimension']}, "
        f"index={metadata['faiss_index']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
