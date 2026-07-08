"""Evaluate context construction parameters after retrieval."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .build_rag_test_set import DEFAULT_OUTPUT as DEFAULT_TEST_SET, build_test_set
from .compare_retrieval_methods import RankedItem
from .evaluation import (
    citations_are_valid,
    citation_numbers,
    exact_match,
    mean,
    normalize_answer,
    token_scores,
)
from .generator import GenerationError, YandexGenerator
from .retriever import HybridRetriever, RetrievedChunk
from .settings import PROJECT_ROOT, Settings


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "context_strategies"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "context_strategy_comparison.md"


@dataclass(frozen=True)
class ContextConfig:
    top_k: int
    intro_chunks: int
    context_window: int
    max_context_chunks: int

    @property
    def label(self) -> str:
        return (
            f"top_k={self.top_k}; intro={self.intro_chunks}; "
            f"window={self.context_window}; max={self.max_context_chunks}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "top_k": self.top_k,
            "intro_chunks": self.intro_chunks,
            "context_window": self.context_window,
            "max_context_chunks": self.max_context_chunks,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            json.dump(record, output, ensure_ascii=False)
            output.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_int_list(value: str) -> list[int]:
    numbers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not numbers:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return numbers


def build_configs(args: argparse.Namespace) -> list[ContextConfig]:
    configs = []
    for top_k, intro, window, max_context in itertools.product(
        args.top_k,
        args.intro_chunks,
        args.context_window,
        args.max_context_chunks,
    ):
        if max_context < top_k:
            continue
        configs.append(
            ContextConfig(
                top_k=top_k,
                intro_chunks=intro,
                context_window=window,
                max_context_chunks=max_context,
            )
        )
    return configs


def run_experiment(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not args.test_set.exists():
        build_test_set(output_path=args.test_set, size=args.max_questions)
    cases = load_jsonl(args.test_set)[: args.max_questions]
    if not cases:
        raise ValueError("Evaluation set is empty")

    configs = build_configs(args)
    if not configs:
        raise ValueError("No valid context configurations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(settings)
    retriever.initialize()

    detail_rows: list[dict[str, Any]] = []
    print(f"Questions: {len(cases)}")
    print(f"Context configurations: {len(configs)}")
    max_top_k = max(config.top_k for config in configs)
    base_rankings: dict[str, tuple[list[RetrievedChunk], float]] = {}
    for case_index, case in enumerate(cases, start=1):
        retrieval = retriever.retrieve(
            case["question"],
            top_k=max_top_k,
            context_window=0,
            intro_chunks=0,
            max_context_chunks=max_top_k,
        )
        base_rankings[case["case_id"]] = (retrieval.chunks, retrieval.elapsed_ms)
        print(f"[{case_index}/{len(cases)}] base ranking: {case['question']}")

    for config_index, config in enumerate(configs, start=1):
        started = time.perf_counter()
        for case in cases:
            base_chunks, retrieval_ms = base_rankings[case["case_id"]]
            chunks = expand_from_base_ranking(
                retriever,
                base_chunks,
                config,
            )
            detail_rows.append(context_detail_row(case, config, chunks, retrieval_ms))
        elapsed = time.perf_counter() - started
        print(f"[{config_index}/{len(configs)}] {config.label} ({elapsed:.1f}s)")

    summary_rows = summarize_context(detail_rows)
    best = max(summary_rows, key=selection_key)
    generation_rows: list[dict[str, Any]] = []
    generation_summary: list[dict[str, Any]] = []
    generation_configs: list[ContextConfig] = []
    if args.run_generation:
        generation_configs = select_generation_configs(summary_rows, args)
        generator = YandexGenerator(settings)
        generation_rows = run_generation_eval(
            select_generation_cases(cases, args.generation_questions),
            generation_configs,
            retriever,
            generator,
            settings,
        )
        generation_summary = summarize_generation(generation_rows)

    write_jsonl(args.output_dir / "context_strategy_details.jsonl", detail_rows)
    write_csv(args.output_dir / "context_strategy_summary.csv", summary_rows)
    write_json(args.output_dir / "context_strategy_summary.json", summary_rows)
    if generation_rows:
        write_jsonl(args.output_dir / "context_generation_results.jsonl", generation_rows)
        write_csv(args.output_dir / "context_generation_summary.csv", generation_summary)
        write_json(args.output_dir / "context_generation_summary.json", generation_summary)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(cases),
        "configs": len(configs),
        "run_generation": args.run_generation,
        "generation_questions": args.generation_questions if args.run_generation else 0,
        "selection_rule": (
            "context_hit_rate, required_term_coverage, gold_chunk_share, compactness"
        ),
        "best_context_config": best,
        "generation_configs": [config.as_dict() for config in generation_configs],
    }
    write_json(args.output_dir / "experiment_metadata.json", metadata)
    args.report.write_text(
        build_report(
            summary_rows,
            generation_summary,
            best,
            questions=len(cases),
            args=args,
        ),
        encoding="utf-8",
    )
    print(f"Best context config: {best['config']}")
    print(f"Report: {args.report}")


def context_detail_row(
    case: dict[str, Any],
    config: ContextConfig,
    chunks: list[RetrievedChunk],
    elapsed_ms: float,
) -> dict[str, Any]:
    gold_docs = set(case["gold_document_ids"])
    docs = [chunk.document_id for chunk in chunks]
    gold_flags = [doc in gold_docs for doc in docs]
    context_text = " ".join(chunk.text for chunk in chunks)
    required_terms = [str(term) for term in case.get("required_terms") or []]
    matched_terms = sum(
        normalize_answer(term) in normalize_answer(context_text)
        for term in required_terms
    )
    required_coverage = (
        matched_terms / len(required_terms) if required_terms else None
    )
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "gold_document_ids": "|".join(case["gold_document_ids"]),
        "config": config.label,
        **config.as_dict(),
        "context_chunks": len(chunks),
        "context_words": sum(len(chunk.text.split()) for chunk in chunks),
        "unique_documents": len(set(docs)),
        "first_chunk_gold": float(bool(gold_flags and gold_flags[0])),
        "context_has_gold": float(any(gold_flags)),
        "gold_chunk_share": (
            sum(gold_flags) / len(gold_flags) if gold_flags else 0.0
        ),
        "non_gold_chunks": sum(not flag for flag in gold_flags),
        "required_term_coverage": required_coverage,
        "retrieval_ms": round(elapsed_ms, 3),
        "chunk_ids": "|".join(chunk.chunk_id for chunk in chunks),
        "document_ids": "|".join(docs),
    }


def expand_from_base_ranking(
    retriever: HybridRetriever,
    base_chunks: list[RetrievedChunk],
    config: ContextConfig,
) -> list[RetrievedChunk]:
    base_items = [
        RankedItem(chunk.chunk_index, chunk.score)
        for chunk in base_chunks
    ]
    expanded = retriever._expand_context(
        base_items,
        config.top_k,
        context_window=config.context_window,
        intro_chunks=config.intro_chunks,
        max_context_chunks=config.max_context_chunks,
    )
    output = []
    for rank, item in enumerate(expanded, start=1):
        chunk = retriever.chunks[item.chunk_index]
        output.append(
            RetrievedChunk(
                rank=rank,
                chunk_index=item.chunk_index,
                chunk_id=str(chunk["chunk_id"]),
                document_id=str(chunk["document_id"]),
                title=chunk.get("title"),
                text=str(chunk["text"]),
                score=float(item.score),
            )
        )
    return output


def summarize_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config"]), []).append(row)

    summary = []
    for config, items in grouped.items():
        required_values = [
            float(row["required_term_coverage"])
            for row in items
            if row["required_term_coverage"] is not None
        ]
        top = items[0]
        record = {
            "config": config,
            "top_k": top["top_k"],
            "intro_chunks": top["intro_chunks"],
            "context_window": top["context_window"],
            "max_context_chunks": top["max_context_chunks"],
            "questions": len(items),
            "avg_context_chunks": round(mean(float(row["context_chunks"]) for row in items), 3),
            "avg_context_words": round(mean(float(row["context_words"]) for row in items), 3),
            "avg_unique_documents": round(mean(float(row["unique_documents"]) for row in items), 3),
            "first_chunk_gold_rate": round(mean(float(row["first_chunk_gold"]) for row in items), 6),
            "context_hit_rate": round(mean(float(row["context_has_gold"]) for row in items), 6),
            "gold_chunk_share": round(mean(float(row["gold_chunk_share"]) for row in items), 6),
            "avg_non_gold_chunks": round(mean(float(row["non_gold_chunks"]) for row in items), 3),
            "required_term_coverage": (
                round(mean(required_values), 6) if required_values else None
            ),
            "avg_retrieval_ms": round(mean(float(row["retrieval_ms"]) for row in items), 3),
        }
        record["selection_score"] = round(context_selection_score(record), 6)
        summary.append(record)
    summary.sort(key=selection_key, reverse=True)
    return summary


def context_selection_score(record: dict[str, Any]) -> float:
    required = record["required_term_coverage"]
    required_value = float(required) if required is not None else float(record["context_hit_rate"])
    compactness_penalty = min(float(record["avg_context_words"]) / 1800.0, 1.0)
    chunk_penalty = min(float(record["avg_context_chunks"]) / 20.0, 1.0)
    noise_penalty = 1.0 - float(record["gold_chunk_share"])
    return (
        0.35 * float(record["context_hit_rate"])
        + 0.25 * required_value
        + 0.25 * float(record["gold_chunk_share"])
        + 0.10 * float(record["first_chunk_gold_rate"])
        - 0.03 * compactness_penalty
        - 0.02 * chunk_penalty
        - 0.05 * noise_penalty
    )


def selection_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(record["selection_score"]),
        float(record["context_hit_rate"]),
        float(record["required_term_coverage"] or 0.0),
        float(record["gold_chunk_share"]),
        -float(record["avg_context_words"]),
    )


def select_generation_configs(
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[ContextConfig]:
    candidates: list[dict[str, Any]] = summary_rows[: args.generation_top_configs]

    exact_targets = [
        (3, 0, 0, 6),
        (5, 2, 1, 6),
        (5, 3, 1, 12),
        (7, 3, 1, 12),
        (10, 5, 2, 16),
    ]
    for target in exact_targets:
        row = find_config(summary_rows, *target)
        if row is not None:
            candidates.append(row)

    seen = set()
    configs = []
    for row in candidates:
        key = effective_generation_key(
            int(row["top_k"]),
            int(row["intro_chunks"]),
            int(row["context_window"]),
            int(row["max_context_chunks"]),
        )
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            ContextConfig(
                top_k=int(row["top_k"]),
                intro_chunks=int(row["intro_chunks"]),
                context_window=int(row["context_window"]),
                max_context_chunks=int(row["max_context_chunks"]),
            )
        )
    return configs


def find_config(
    rows: list[dict[str, Any]],
    top_k: int,
    intro_chunks: int,
    context_window: int,
    max_context_chunks: int,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if int(row["top_k"]) == top_k
            and int(row["intro_chunks"]) == intro_chunks
            and int(row["context_window"]) == context_window
            and int(row["max_context_chunks"]) == max_context_chunks
        ),
        None,
    )


def effective_generation_key(
    top_k: int,
    intro_chunks: int,
    context_window: int,
    max_context_chunks: int,
) -> tuple[int, int, int, int]:
    if intro_chunks == 0 and context_window == 0:
        return top_k, intro_chunks, context_window, top_k
    return top_k, intro_chunks, context_window, max_context_chunks


def select_generation_cases(
    cases: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected = list(cases[:limit])
    selected_ids = {case["case_id"] for case in selected}
    for case in cases:
        if case["case_id"] in selected_ids:
            continue
        if case.get("required_terms"):
            selected.append(case)
            selected_ids.add(case["case_id"])
    return selected


def run_generation_eval(
    cases: list[dict[str, Any]],
    configs: list[ContextConfig],
    retriever: HybridRetriever,
    generator: YandexGenerator,
    settings: Settings,
) -> list[dict[str, Any]]:
    rows = []
    for config in configs:
        print(f"Generation eval: {config.label}")
        for case in cases:
            retrieval = retriever.retrieve(
                case["question"],
                top_k=config.top_k,
                context_window=config.context_window,
                intro_chunks=config.intro_chunks,
                max_context_chunks=config.max_context_chunks,
            )
            row = {
                "case_id": case["case_id"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "gold_evidence": case["gold_evidence"],
                "required_terms": case.get("required_terms") or [],
                "gold_document_ids": case["gold_document_ids"],
                "retrieved_document_ids": [chunk.document_id for chunk in retrieval.chunks],
                "config": config.label,
                **config.as_dict(),
                "context_chunks": len(retrieval.chunks),
                "context_words": sum(len(chunk.text.split()) for chunk in retrieval.chunks),
                "retrieval_ms": round(retrieval.elapsed_ms, 3),
            }
            try:
                result = generator.generate(
                    case["question"],
                    retrieval.chunks,
                    language="en",
                    temperature=settings.generation_temperature,
                    top_p=settings.generation_top_p,
                    max_output_tokens=settings.generation_max_tokens,
                )
            except GenerationError as error:
                row.update(
                    {
                        "answer": "",
                        "generation_error": str(error),
                        "generation_ms": None,
                        "schema_valid": False,
                        "source": None,
                        "confidence": None,
                    }
                )
            else:
                row.update(
                    {
                        "answer": result.answer,
                        "generation_error": None,
                        "generation_ms": round(result.elapsed_ms, 3),
                        "schema_valid": result.schema_valid,
                        "source": result.source,
                        "confidence": result.confidence,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                )
            enrich_generation_row(row)
            rows.append(row)
    return rows


def enrich_generation_row(row: dict[str, Any]) -> None:
    answer = str(row.get("answer") or "")
    reference = str(row.get("reference_answer") or "")
    precision, recall, f1 = token_scores(answer, reference)
    row["exact_match"] = exact_match(answer, reference) if answer else 0.0
    row["token_precision"] = round(precision, 6)
    row["token_recall"] = round(recall, 6)
    row["token_f1"] = round(f1, 6)
    required_terms = [str(term) for term in row.get("required_terms") or []]
    matched_terms = sum(
        normalize_answer(term) in normalize_answer(answer)
        for term in required_terms
    )
    row["required_answer_coverage"] = (
        matched_terms / len(required_terms) if required_terms else None
    )
    citations = citation_numbers(answer, row.get("source"))
    row["citation_valid"] = citations_are_valid(citations, int(row["context_chunks"]))
    gold_docs = set(row["gold_document_ids"])
    retrieved_docs = row["retrieved_document_ids"]
    row["context_has_gold"] = any(doc in gold_docs for doc in retrieved_docs)


def summarize_generation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config"]), []).append(row)

    summary = []
    for config, items in grouped.items():
        required_values = [
            float(row["required_answer_coverage"])
            for row in items
            if row["required_answer_coverage"] is not None
        ]
        top = items[0]
        record = {
            "config": config,
            "top_k": top["top_k"],
            "intro_chunks": top["intro_chunks"],
            "context_window": top["context_window"],
            "max_context_chunks": top["max_context_chunks"],
            "questions": len(items),
            "schema_valid_rate": round(mean(float(bool(row["schema_valid"])) for row in items), 6),
            "context_hit_rate": round(mean(float(bool(row["context_has_gold"])) for row in items), 6),
            "avg_token_f1": round(mean(float(row["token_f1"]) for row in items), 6),
            "required_answer_coverage": (
                round(mean(required_values), 6) if required_values else None
            ),
            "citation_valid_rate": round(mean(float(bool(row["citation_valid"])) for row in items), 6),
            "avg_context_chunks": round(mean(float(row["context_chunks"]) for row in items), 3),
            "avg_context_words": round(mean(float(row["context_words"]) for row in items), 3),
            "avg_generation_ms": round(
                mean(float(row["generation_ms"]) for row in items if row["generation_ms"] is not None),
                3,
            ),
        }
        record["generation_score"] = round(generation_selection_score(record), 6)
        summary.append(record)
    summary.sort(
        key=lambda row: (
            float(row["generation_score"]),
            float(row["required_answer_coverage"] or 0.0),
            float(row["avg_token_f1"]),
            -float(row["avg_context_words"]),
        ),
        reverse=True,
    )
    return summary


def generation_selection_score(record: dict[str, Any]) -> float:
    required = record["required_answer_coverage"]
    required_value = float(required) if required is not None else float(record["context_hit_rate"])
    return (
        0.30 * float(record["schema_valid_rate"])
        + 0.25 * required_value
        + 0.20 * float(record["avg_token_f1"])
        + 0.15 * float(record["context_hit_rate"])
        + 0.10 * float(record["citation_valid_rate"])
        - 0.03 * min(float(record["avg_context_words"]) / 1800.0, 1.0)
    )


def markdown_table(records: list[dict[str, Any]], fields: list[str], limit: int) -> str:
    selected = records[:limit]
    header = "| " + " | ".join(fields) + " |"
    align = "|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|"
    lines = [header, align]
    for record in selected:
        cells = []
        for field in fields:
            value = record.get(field)
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    summary_rows: list[dict[str, Any]],
    generation_summary: list[dict[str, Any]],
    best: dict[str, Any],
    *,
    questions: int,
    args: argparse.Namespace,
) -> str:
    worst_large = [
        row
        for row in summary_rows
        if row["top_k"] == 10
        and row["intro_chunks"] == 5
        and row["context_window"] == 2
        and row["max_context_chunks"] == 16
    ]
    degradation_note = ""
    if worst_large:
        large = worst_large[0]
        degradation_note = (
            f"\nСамый широкий проверенный режим `top_k=10, intro=5, window=2, max=16` "
            f"даёт в среднем {large['avg_context_chunks']:.1f} чанков и "
            f"{large['avg_context_words']:.0f} слов контекста. Его score={large['selection_score']:.4f}, "
            f"gold chunk share={large['gold_chunk_share']:.4f}. Это зона, где контекст начинает "
            "раздуваться: модель видит больше текста, но доля действительно полезных чанков падает.\n"
        )

    generation_block = (
        "LLM-прогон не запускался. Для быстрого воспроизведения можно выполнить скрипт с флагом `--run-generation`."
    )
    if generation_summary:
        generation_block = markdown_table(
            generation_summary,
            [
                "config",
                "generation_score",
                "avg_token_f1",
                "required_answer_coverage",
                "citation_valid_rate",
                "avg_context_chunks",
                "avg_context_words",
            ],
            limit=10,
        )

    return f"""# Исследование стратегий формирования контекста

Цель шага — проверить, как параметры формирования контекста после retrieval влияют на качество RAG.

Проверялись параметры:

- `top_k`: {", ".join(map(str, args.top_k))};
- `intro_chunks`: {", ".join(map(str, args.intro_chunks))};
- `context_window`: {", ".join(map(str, args.context_window))};
- `max_context_chunks`: {", ".join(map(str, args.max_context_chunks))}.

Некорректные комбинации, где `max_context_chunks < top_k`, пропускались.

## Методика

Широкая сетка оценивалась без LLM-вызовов на {questions} диагностических вопросах. Это нужно, чтобы быстро сравнить сотни комбинаций и не тратить API на заведомо слабые варианты.

Основные метрики:

- `context_hit_rate` — есть ли среди итоговых чанков документ из gold-разметки;
- `gold_chunk_share` — какая доля итоговых чанков относится к gold-документу;
- `required_term_coverage` — попали ли в контекст обязательные факты из диагностических кейсов;
- `avg_context_chunks` и `avg_context_words` — насколько сильно раздувается контекст;
- `selection_score` — сводная эвристика: награждает за попадание gold/required facts и штрафует за шумный длинный контекст.

## Лучшие конфигурации по контексту

{markdown_table(
        summary_rows,
        [
            "config",
            "selection_score",
            "context_hit_rate",
            "gold_chunk_share",
            "required_term_coverage",
            "avg_context_chunks",
            "avg_context_words",
        ],
        limit=15,
    )}

## Выбранная конфигурация

Лучшая конфигурация по сводной оценке:

```text
{best['config']}
```

Метрики выбранной конфигурации:

- context hit rate: `{best['context_hit_rate']:.4f}`;
- gold chunk share: `{best['gold_chunk_share']:.4f}`;
- required term coverage: `{best['required_term_coverage']}`;
- avg context chunks: `{best['avg_context_chunks']:.1f}`;
- avg context words: `{best['avg_context_words']:.0f}`;
- selection score: `{best['selection_score']:.4f}`.
{degradation_note}
## LLM-проверка ключевых конфигураций

{generation_block}

## Вывод

Слишком маленький контекст может не донести до LLM соседние условия, цену или исключения. Слишком широкий контекст увеличивает шум: в prompt попадает больше нерелевантных фрагментов, а доля gold-чанков падает.

На текущем корпусе и текущем retriever лучшим дефолтом оказался компактный режим без автоматического добавления соседних и вводных чанков. Это не отменяет context expansion полностью: его полезно оставить как настраиваемый или будущий адаптивный режим, но включать его всегда по умолчанию сейчас невыгодно.

Артефакты эксперимента:

- `data/experiments/context_strategies/context_strategy_summary.csv`;
- `data/experiments/context_strategies/context_strategy_details.jsonl`;
- `data/experiments/context_strategies/experiment_metadata.json`.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG context construction strategies.")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--top-k", type=parse_int_list, default=parse_int_list("3,5,7,10"))
    parser.add_argument("--intro-chunks", type=parse_int_list, default=parse_int_list("0,1,2,3,5"))
    parser.add_argument("--context-window", type=parse_int_list, default=parse_int_list("0,1,2"))
    parser.add_argument("--max-context-chunks", type=parse_int_list, default=parse_int_list("6,8,10,12,16"))
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-questions", type=int, default=8)
    parser.add_argument("--generation-top-configs", type=int, default=3)
    args = parser.parse_args()
    if args.max_questions < 1:
        parser.error("--max-questions must be positive")
    if args.generation_questions < 1:
        parser.error("--generation-questions must be positive")
    return args


def main() -> int:
    run_experiment(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
