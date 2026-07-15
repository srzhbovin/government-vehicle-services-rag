#!/usr/bin/env python3
"""Compare embedding models with the same exact FAISS retrieval protocol."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .compare_retrieval_methods import ROOT, parse_args, run_experiment


DEFAULT_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "experiments" / "embedding_models"
DEFAULT_REPORT = ROOT / "reports" / "embedding_model_comparison.md"


def model_slug(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower().replace("_", "-")


def compare_models(
    models: Sequence[str],
    *,
    output_dir: Path,
    report_path: Path,
    device: str | None = None,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for model in models:
        slug = model_slug(model)
        model_dir = output_dir / slug
        arguments = [
            "--embedding-model",
            model,
            "--output-dir",
            str(model_dir),
            "--index-dir",
            str(model_dir / "index"),
            "--report",
            str(model_dir / "retrieval_report.md"),
            "--skip-test",
        ]
        if device:
            arguments.extend(["--device", device])
        records = run_experiment(parse_args(arguments))
        faiss_record = next(
            record
            for record in records
            if record["split"] == "validation" and record["method"] == "faiss_flat"
        )
        summary.append(
            {
                "model": model,
                "dimension": faiss_record["dimension"],
                "build_ms": round(float(faiss_record["build_ms"]), 3),
                "search_ms_per_query": round(
                    float(faiss_record["search_ms_per_query"]), 6
                ),
                "recall@1": round(float(faiss_record["recall@1"]), 6),
                "recall@5": round(float(faiss_record["recall@5"]), 6),
                "recall@10": round(float(faiss_record["recall@10"]), 6),
                "mrr@10": round(float(faiss_record["mrr@10"]), 6),
            }
        )

    selected = max(
        summary,
        key=lambda row: (
            float(row["recall@5"]),
            float(row["mrr@10"]),
            -float(row["search_ms_per_query"]),
        ),
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "selection_rule": "Recall@5, then MRR@10, then search latency",
        "selected_model": selected["model"],
        "models": list(models),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(summary[0])
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    rows = "\n".join(
        "| {model} | {dimension} | {build_ms:.1f} | {search_ms_per_query:.3f} | "
        "{recall_at_1:.4f} | {recall_at_5:.4f} | {recall_at_10:.4f} | {mrr_at_10:.4f} |".format(
            model=row["model"],
            dimension=int(row["dimension"]),
            build_ms=float(row["build_ms"]),
            search_ms_per_query=float(row["search_ms_per_query"]),
            recall_at_1=float(row["recall@1"]),
            recall_at_5=float(row["recall@5"]),
            recall_at_10=float(row["recall@10"]),
            mrr_at_10=float(row["mrr@10"]),
        )
        for row in summary
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# Сравнение embedding-моделей

## Методика

Обе модели проверяются на одинаковых token chunks `120/0` и validation-вопросах
MultiDoc2Dial DMV. Для каждой модели строятся нормализованные эмбеддинги и точный
`FAISS IndexFlatIP`. Меняется только embedding-модель. Основная метрика выбора —
document-level Recall@5; test-набор при выборе не используется.

## Результаты

| Модель | Размерность | Построение, мс | Поиск, мс/запрос | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Вывод

По правилу `Recall@5 → MRR@10 → скорость` выбрана
`{selected['model']}`. Вывод относится к текущему англоязычному DMV-корпусу и
точному FAISS-поиску; при смене корпуса сравнение следует повторить.

## Воспроизведение

```powershell
python -m src.rag_pipeline.compare_embedding_models
```

Компактные результаты и метаданные сохраняются в `data/experiments/embedding_models/`.
Отдельные FAISS-индексы, эмбеддинги и per-query строки также создаются локально, но
игнорируются Git из-за размера и не перезаписывают рабочий индекс RAG.
""",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if len(args.models) < 2:
        parser.error("Укажите минимум две embedding-модели")
    compare_models(
        args.models,
        output_dir=args.output_dir,
        report_path=args.report,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
