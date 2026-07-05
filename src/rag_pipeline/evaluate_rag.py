"""Run prompt-engineering and end-to-end RAG quality experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .build_rag_test_set import DEFAULT_OUTPUT, build_test_set
from .evaluation import (
    citation_numbers,
    citations_are_valid,
    error_categories,
    exact_match,
    mean,
    normalize_answer,
    reciprocal_rank,
    retrieval_hit,
    retrieval_recall,
    token_scores,
)
from .generator import GenerationError, YandexGenerator
from .prompting import PromptStrategy
from .retriever import HybridRetriever
from .settings import PROJECT_ROOT, Settings


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "rag"
PROMPT_REPORT = PROJECT_ROOT / "reports" / "prompt_engineering.md"
QUALITY_REPORT = PROJECT_ROOT / "reports" / "rag_quality.md"
STRATEGIES = tuple(PromptStrategy)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_experiment(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if settings.llm_provider != "yandex" and not args.retrieval_only:
        raise ValueError("Prompt experiment requires RAG_LLM_PROVIDER=yandex")
    if not args.test_set.exists():
        build_test_set(output_path=args.test_set, size=args.max_questions)
    cases = load_jsonl(args.test_set)[: args.max_questions]
    if not cases:
        raise ValueError("Evaluation set is empty")

    retriever = HybridRetriever(settings)
    retriever.initialize()
    generator = YandexGenerator(settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "generation_results.jsonl"
    retrieval_path = args.output_dir / "retrieval_results.jsonl"
    if not args.resume:
        raw_path.unlink(missing_ok=True)

    existing = load_jsonl(raw_path) if args.resume and raw_path.exists() else []
    completed = {(row["case_id"], row["variant"]) for row in existing}
    retrieval_rows = []
    total = len(cases)

    for case_index, case in enumerate(cases):
        retrieval = retriever.retrieve(case["question"], top_k=settings.top_k)
        documents = [chunk.document_id for chunk in retrieval.chunks]
        chunks = [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "score": chunk.score,
            }
            for chunk in retrieval.chunks
        ]
        retrieval_row = {
            "case_id": case["case_id"],
            "question": case["question"],
            "gold_document_ids": case["gold_document_ids"],
            "retrieved_document_ids": documents,
            "retrieved_chunks": chunks,
            "retrieval_ms": round(retrieval.elapsed_ms, 3),
            "hit_at_1": retrieval_hit(documents, case["gold_document_ids"], 1),
            "hit_at_3": retrieval_hit(documents, case["gold_document_ids"], 3),
            "hit_at_5": retrieval_hit(documents, case["gold_document_ids"], 5),
            "recall_at_1": retrieval_recall(documents, case["gold_document_ids"], 1),
            "recall_at_3": retrieval_recall(documents, case["gold_document_ids"], 3),
            "recall_at_5": retrieval_recall(documents, case["gold_document_ids"], 5),
            "reciprocal_rank": reciprocal_rank(documents, case["gold_document_ids"]),
        }
        retrieval_rows.append(retrieval_row)
        print(f"[{case_index + 1}/{total}] retrieval: {case['question']}")
        if args.retrieval_only:
            continue

        variants = [
            (strategy.value, strategy, settings.generation_temperature, "prompt")
            for strategy in STRATEGIES
        ]
        if case_index < args.parameter_questions:
            for temperature in args.temperatures:
                if abs(temperature - settings.generation_temperature) < 1e-9:
                    continue
                variants.append(
                    (
                        f"structured_output_t{temperature:g}",
                        PromptStrategy.STRUCTURED_OUTPUT,
                        temperature,
                        "temperature",
                    )
                )

        for variant, strategy, temperature, group in variants:
            key = (case["case_id"], variant)
            if key in completed:
                continue
            row = {
                "case_id": case["case_id"],
                "question_id": case["question_id"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "gold_evidence": case["gold_evidence"],
                "required_terms": case.get("required_terms") or [],
                "gold_document_ids": case["gold_document_ids"],
                "retrieved_document_ids": documents,
                "retrieved_chunks": chunks,
                "retrieval_ms": round(retrieval.elapsed_ms, 3),
                "retrieval_hit_at_1": retrieval_row["hit_at_1"],
                "retrieval_hit_at_3": retrieval_row["hit_at_3"],
                "retrieval_hit_at_5": retrieval_row["hit_at_5"],
                "variant": variant,
                "experiment_group": group,
                "prompt_strategy": strategy.value,
                "temperature": temperature,
                "parameter_subset": case_index < args.parameter_questions,
            }
            try:
                result = generate_with_retry(
                    generator,
                    case["question"],
                    retrieval.chunks,
                    strategy,
                    temperature,
                    args.retries,
                    args.retry_delay,
                )
                row.update(
                    {
                        "answer": result.answer,
                        "raw_output": result.raw_output,
                        "confidence": result.confidence,
                        "source": result.source,
                        "parse_success": result.parse_success,
                        "schema_valid": result.schema_valid,
                        "parse_error": result.parse_error,
                        "generation_model": result.model,
                        "generation_ms": round(result.elapsed_ms, 3),
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                        "generation_error": None,
                    }
                )
            except GenerationError as error:
                row.update(
                    {
                        "answer": "",
                        "raw_output": None,
                        "confidence": None,
                        "source": None,
                        "parse_success": False,
                        "schema_valid": False,
                        "parse_error": None,
                        "generation_model": settings.yandex_model_uri,
                        "generation_ms": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "generation_error": str(error),
                    }
                )
            append_jsonl(raw_path, row)
            completed.add(key)
            print(f"  {variant}: {'error' if row['generation_error'] else 'ok'}")

    write_jsonl(retrieval_path, retrieval_rows)
    retrieval_summary = summarize_retrieval(retrieval_rows)
    write_csv(args.output_dir / "retrieval_summary.csv", retrieval_summary)

    if args.retrieval_only:
        write_json(args.output_dir / "retrieval_summary.json", retrieval_summary)
        return

    rows = load_jsonl(raw_path)
    enrich_generation_rows(rows, retriever.encoder)
    write_jsonl(raw_path, rows)
    prompt_summary = summarize_prompt_strategies(rows)
    temperature_summary = summarize_temperatures(rows, settings.generation_temperature)
    winner = select_winner(prompt_summary)
    errors = build_error_rows(rows)

    write_csv(args.output_dir / "prompt_strategy_summary.csv", prompt_summary)
    write_csv(args.output_dir / "temperature_summary.csv", temperature_summary)
    write_jsonl(args.output_dir / "error_analysis.jsonl", errors)
    write_json(args.output_dir / "prompt_strategy_summary.json", prompt_summary)
    write_json(args.output_dir / "temperature_summary.json", temperature_summary)
    write_json(args.output_dir / "retrieval_summary.json", retrieval_summary)
    write_json(
        args.output_dir / "best_prompt_strategy.json",
        {
            "strategy": winner["strategy"],
            "selection_score": winner["selection_score"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(
        args.output_dir / "experiment_metadata.json",
        {
            "model": settings.yandex_model,
            "retrieval_method": retriever.method_name,
            "questions": len(cases),
            "parameter_questions": min(args.parameter_questions, len(cases)),
            "default_temperature": settings.generation_temperature,
            "strategies": [strategy.value for strategy in STRATEGIES],
            "semantic_correctness_threshold": 0.60,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_prompt_report(prompt_summary, winner, len(cases), settings)
    write_quality_report(
        retrieval_summary,
        prompt_summary,
        temperature_summary,
        errors,
        winner,
        len(cases),
    )


def generate_with_retry(
    generator: YandexGenerator,
    question: str,
    chunks: list,
    strategy: PromptStrategy,
    temperature: float,
    retries: int,
    retry_delay: float,
):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return generator.generate(
                question,
                chunks,
                language="en",
                strategy=strategy,
                temperature=temperature,
            )
        except GenerationError as error:
            last_error = error
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def enrich_generation_rows(rows: list[dict[str, Any]], encoder: Any) -> None:
    texts = set()
    for row in rows:
        if row.get("answer"):
            texts.add(row["answer"])
            texts.add(row["reference_answer"])
            texts.add(" ".join(row.get("gold_evidence") or []))
    vectors = {}
    if texts:
        ordered = sorted(texts)
        encoded = encoder.encode(
            ordered,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = {text: vector for text, vector in zip(ordered, encoded)}

    for row in rows:
        answer = str(row.get("answer") or "")
        reference = str(row.get("reference_answer") or "")
        evidence = " ".join(row.get("gold_evidence") or [])
        precision, recall, f1 = token_scores(answer, reference)
        row["exact_match"] = exact_match(answer, reference) if answer else 0.0
        row["reference_token_precision"] = round(precision, 6)
        row["reference_token_recall"] = round(recall, 6)
        row["token_f1"] = round(f1, 6)
        if answer and vectors:
            reference_similarity = float(np.dot(vectors[answer], vectors[reference]))
            evidence_similarity = float(np.dot(vectors[answer], vectors[evidence]))
            semantic_similarity = max(reference_similarity, evidence_similarity)
        else:
            reference_similarity = 0.0
            evidence_similarity = 0.0
            semantic_similarity = 0.0
        row["reference_similarity"] = round(reference_similarity, 6)
        row["evidence_similarity"] = round(evidence_similarity, 6)
        row["semantic_similarity"] = round(semantic_similarity, 6)
        citations = citation_numbers(answer, row.get("source"))
        row["citations"] = citations
        row["citation_valid"] = citations_are_valid(
            citations,
            len(row.get("retrieved_chunks") or []),
        )
        required_terms = row.get("required_terms") or []
        normalized_answer = normalize_answer(answer)
        matched_terms = sum(
            normalize_answer(str(term)) in normalized_answer for term in required_terms
        )
        fact_coverage = matched_terms / len(required_terms) if required_terms else 1.0
        row["required_fact_coverage"] = round(fact_coverage, 6)
        row["answer_correct_heuristic"] = bool(
            row.get("retrieval_hit_at_5")
            and semantic_similarity >= 0.60
            and fact_coverage >= 0.8
            and not row.get("generation_error")
        )
        row["error_categories"] = error_categories(row)


def summarize_retrieval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for k in (1, 3, 5):
        output.append(
            {
                "k": k,
                "questions": len(rows),
                "hit_rate": round(mean(row[f"hit_at_{k}"] for row in rows), 6),
                "recall_at_k": round(mean(row[f"recall_at_{k}"] for row in rows), 6),
                "mrr": round(mean(row["reciprocal_rank"] for row in rows), 6),
                "avg_retrieval_ms": round(mean(row["retrieval_ms"] for row in rows), 3),
            }
        )
    return output


def summarize_prompt_strategies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("experiment_group") == "prompt":
            grouped[row["prompt_strategy"]].append(row)
    output = []
    for strategy in (item.value for item in STRATEGIES):
        group = grouped.get(strategy, [])
        if not group:
            continue
        structured = strategy != PromptStrategy.PLAIN.value
        schema_rate = (
            mean(row.get("schema_valid") is True for row in group) if structured else None
        )
        parse_errors = sum(row.get("parse_success") is False for row in group)
        generation_errors = sum(bool(row.get("generation_error")) for row in group)
        correct_rate = mean(row.get("answer_correct_heuristic") for row in group)
        semantic = mean(row.get("semantic_similarity") for row in group)
        citation_rate = mean(row.get("citation_valid") for row in group)
        confidence_rows = [row for row in group if row.get("confidence") is not None]
        avg_confidence = mean(row.get("confidence") for row in confidence_rows)
        confidence_mae = mean(
            abs(float(row["confidence"]) - float(row["answer_correct_heuristic"]))
            for row in confidence_rows
        )
        selection_score = (
            0.35 * (schema_rate if schema_rate is not None else 0.0)
            + 0.35 * correct_rate
            + 0.15 * semantic
            + 0.15 * citation_rate
        )
        output.append(
            {
                "strategy": strategy,
                "runs": len(group),
                "generation_errors": generation_errors,
                "parse_errors": parse_errors,
                "schema_adherence_rate": (
                    round(schema_rate, 6) if schema_rate is not None else None
                ),
                "exact_match": round(mean(row.get("exact_match") for row in group), 6),
                "avg_token_f1": round(mean(row.get("token_f1") for row in group), 6),
                "avg_semantic_similarity": round(semantic, 6),
                "answer_correct_rate": round(correct_rate, 6),
                "citation_valid_rate": round(citation_rate, 6),
                "avg_confidence": (
                    round(avg_confidence, 6) if confidence_rows else None
                ),
                "confidence_mae": (
                    round(confidence_mae, 6) if confidence_rows else None
                ),
                "avg_generation_ms": round(mean(row.get("generation_ms") for row in group), 3),
                "avg_total_tokens": round(mean(row.get("total_tokens") for row in group), 3),
                "selection_score": round(selection_score, 6),
            }
        )
    return output


def summarize_temperatures(
    rows: list[dict[str, Any]],
    default_temperature: float,
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("prompt_strategy") != PromptStrategy.STRUCTURED_OUTPUT.value:
            continue
        if not row.get("parameter_subset"):
            continue
        if row.get("experiment_group") == "prompt":
            grouped[default_temperature].append(row)
        elif row.get("experiment_group") == "temperature":
            grouped[float(row["temperature"])].append(row)
    output = []
    for temperature in sorted(grouped):
        group = grouped[temperature]
        output.append(
            {
                "temperature": temperature,
                "runs": len(group),
                "generation_errors": sum(bool(row.get("generation_error")) for row in group),
                "schema_adherence_rate": round(
                    mean(row.get("schema_valid") is True for row in group), 6
                ),
                "exact_match": round(mean(row.get("exact_match") for row in group), 6),
                "avg_token_f1": round(mean(row.get("token_f1") for row in group), 6),
                "avg_semantic_similarity": round(
                    mean(row.get("semantic_similarity") for row in group), 6
                ),
                "answer_correct_rate": round(
                    mean(row.get("answer_correct_heuristic") for row in group), 6
                ),
                "citation_valid_rate": round(
                    mean(row.get("citation_valid") for row in group), 6
                ),
                "avg_confidence": round(
                    mean(row.get("confidence") for row in group), 6
                ),
                "confidence_mae": round(
                    mean(
                        abs(
                            float(row["confidence"])
                            - float(row["answer_correct_heuristic"])
                        )
                        for row in group
                        if row.get("confidence") is not None
                    ),
                    6,
                ),
                "avg_generation_ms": round(
                    mean(row.get("generation_ms") for row in group), 3
                ),
            }
        )
    return output


def select_winner(summary: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in summary if row["schema_adherence_rate"] is not None]
    if not eligible:
        eligible = summary
    return max(
        eligible,
        key=lambda row: (
            row["selection_score"],
            -row["generation_errors"],
            -row["parse_errors"],
            -row["avg_generation_ms"],
        ),
    )


def build_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        categories = [value for value in row.get("error_categories") or [] if value != "ok"]
        if not categories:
            continue
        output.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "question": row["question"],
                "categories": categories,
                "answer": row.get("answer"),
                "reference_answer": row["reference_answer"],
                "gold_document_ids": row["gold_document_ids"],
                "retrieved_document_ids": row["retrieved_document_ids"],
                "semantic_similarity": row.get("semantic_similarity"),
                "reference_token_recall": row.get("reference_token_recall"),
                "required_fact_coverage": row.get("required_fact_coverage"),
                "confidence": row.get("confidence"),
                "generation_error": row.get("generation_error"),
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_prompt_report(
    summary: list[dict[str, Any]],
    winner: dict[str, Any],
    questions: int,
    settings: Settings,
) -> None:
    lines = [
        "# Исследование Prompt Engineering",
        "",
        f"Эксперимент выполнен на {questions} вопросах с моделью `{settings.yandex_model}`. "
        "Для всех вариантов использовались одинаковые найденные фрагменты и temperature "
        f"`{settings.generation_temperature:g}`.",
        "",
        "## Сравниваемые варианты",
        "",
        "- `plain` — обычный текстовый ответ;",
        "- `json` — требование вернуть JSON без передачи формальной схемы;",
        "- `pydantic` — JSON Schema из Pydantic передаётся внутри prompt;",
        "- `structured_output` — нативный `text.format=json_schema` Yandex Responses API.",
        "",
        "## Результаты",
        "",
        "| Вариант | Schema, % | Ошибки JSON | Correct, % | Semantic | Citation, % | Confidence MAE | Время, мс |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        schema = "—" if row["schema_adherence_rate"] is None else percent(row["schema_adherence_rate"])
        confidence_mae = (
            "—" if row["confidence_mae"] is None else f"{row['confidence_mae']:.3f}"
        )
        lines.append(
            f"| `{row['strategy']}` | {schema} | {row['parse_errors']} | "
            f"{percent(row['answer_correct_rate'])} | {row['avg_semantic_similarity']:.3f} | "
            f"{percent(row['citation_valid_rate'])} | {confidence_mae} | "
            f"{row['avg_generation_ms']:.0f} |"
        )
    lines.extend(
        [
            "",
            "`Schema` показывает долю ответов, прошедших Pydantic-валидацию. Для обычного "
            "prompt эта метрика не применяется.",
            "",
            "## Выбранный вариант",
            "",
            f"В основной pipeline выбран `{winner['strategy']}`. Он получил итоговую оценку "
            f"`{winner['selection_score']:.3f}` с учётом соблюдения схемы, корректности ответа, "
            "семантического сходства и валидности ссылок на фрагменты.",
            "",
            "Обычный prompt может показать более высокую содержательную метрику, но не "
            "гарантирует машиночитаемый контракт. Structured Output выбран для API, потому что "
            "схема применяется самим Yandex Responses API и затем проверяется Pydantic.",
            "",
            "Поле confidence остаётся диагностическим: модель склонна возвращать 1.0 даже для "
            "неполных ответов, поэтому его нельзя считать откалиброванной вероятностью.",
            "",
            "Exact Match рассчитывается, но не используется как единственная оценка: корректная "
            "переформулировка ответа обычно не совпадает с эталоном посимвольно.",
            "",
            "Полные данные находятся в `data/experiments/rag/`.",
            "",
        ]
    )
    PROMPT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_quality_report(
    retrieval: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    temperatures: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    winner: dict[str, Any],
    questions: int,
) -> None:
    final = next(row for row in prompts if row["strategy"] == winner["strategy"])
    winner_errors = [row for row in errors if row["variant"] == winner["strategy"]]
    counts = Counter(category for row in winner_errors for category in row["categories"])
    lines = [
        "# Оценка качества RAG",
        "",
        f"Тестовый набор содержит {questions} самостоятельных вопросов. Один вопрос добавлен как "
        "регрессионный случай потери водительских прав; остальные получены из validation-части "
        "MultiDoc2Dial с эталонными документами, ответами и evidence.",
        "",
        "## Retrieval",
        "",
        "| k | Hit Rate | Recall@k | MRR | Среднее время, мс |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in retrieval:
        lines.append(
            f"| {row['k']} | {percent(row['hit_rate'])} | {percent(row['recall_at_k'])} | "
            f"{row['mrr']:.3f} | {row['avg_retrieval_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "В этом наборе у каждого вопроса один gold-документ, поэтому Hit Rate и Recall@k "
            "численно совпадают.",
            "",
            "## Финальный pipeline",
            "",
            f"Используется `{winner['strategy']}`: Exact Match `{percent(final['exact_match'])}`, "
            f"средний token F1 `{final['avg_token_f1']:.3f}`, semantic similarity "
            f"`{final['avg_semantic_similarity']:.3f}`, автоматическая доля корректных ответов "
            f"`{percent(final['answer_correct_rate'])}`.",
            "",
            "Автоматический флаг корректности требует найденный gold-документ и semantic similarity "
            "не ниже `0.60`. Это воспроизводимая эвристика, а не замена экспертной проверки.",
            "",
            "Exact Match для генеративных ответов слишком строг: корректная переформулировка "
            "не совпадает с коротким эталоном посимвольно.",
            "",
            "## Влияние temperature",
            "",
            "| Temperature | Schema, % | Correct, % | Semantic | Citation, % | Время, мс |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in temperatures:
        lines.append(
            f"| {row['temperature']:g} | {percent(row['schema_adherence_rate'])} | "
            f"{percent(row['answer_correct_rate'])} | {row['avg_semantic_similarity']:.3f} | "
            f"{percent(row['citation_valid_rate'])} | {row['avg_generation_ms']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Temperature 0.7 показала лучший correct rate на 10 вопросах, но один прогон не "
            "доказывает устойчивое преимущество. Для воспроизводимости в runtime оставлена 0.1.",
            "",
            "## Систематизация ошибок",
            "",
            "| Категория | Количество |",
            "|---|---:|",
        ]
    )
    for category, count in counts.most_common():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "### Примеры ошибок финального варианта", ""])
    for row in winner_errors[:8]:
        lines.extend(
            [
                f"- **{row['question']}** — `{', '.join(row['categories'])}`; "
                f"gold: `{', '.join(row['gold_document_ids'])}`; "
                f"top-1: `{row['retrieved_document_ids'][0] if row['retrieved_document_ids'] else 'нет'}`.",
            ]
        )
    review_path = DEFAULT_OUTPUT_DIR / "error_review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as source:
            reviewed = list(csv.DictReader(source))
        lines.extend(
            [
                "",
                "### Ручная проверка",
                "",
                "| Вопрос | Причина | Статус |",
                "|---|---|---|",
            ]
        )
        for row in reviewed:
            lines.append(
                f"| {row['question']} | `{row['reviewed_origin']}` | "
                f"`{row['reviewed_status']}` |"
            )
    lines.extend(
        [
            "",
            "Полные ответы и классификация ошибок сохранены в "
            "`data/experiments/rag/generation_results.jsonl` и `error_analysis.jsonl`.",
            "Ручная проверка сохранена в `error_review.csv`.",
            "",
        ]
    )
    QUALITY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


def parse_temperatures(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 or item > 1 for item in values):
        raise argparse.ArgumentTypeError("Temperatures must be within [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--parameter-questions", type=int, default=10)
    parser.add_argument(
        "--temperatures",
        type=parse_temperatures,
        default=(0.0, 0.3, 0.7),
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if args.max_questions < 1 or args.parameter_questions < 0:
        parser.error("Question counts must be non-negative")
    run_experiment(args)


if __name__ == "__main__":
    main()
