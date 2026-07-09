"""Compare LLM backends on the same retrieved RAG contexts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .build_rag_test_set import DEFAULT_OUTPUT, build_test_set
from .evaluate_rag import enrich_generation_rows
from .evaluation import mean
from .generator import GenerationError, YandexGenerator
from .prompting import PromptStrategy
from .retriever import HybridRetriever
from .settings import PROJECT_ROOT, Settings


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "llm_models"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "llm_model_comparison.md"
DEFAULT_MODELS = (
    "yandexgpt-5-lite",
    "yandexgpt-5-pro",
    "yandexgpt-5.1",
    "qwen3.6-35b-a3b",
    "qwen3-235b-a22b-fp8",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "deepseek-v4-flash",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("expected at least one model")
    return models


def run_experiment(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if settings.llm_provider != "yandex":
        raise ValueError("LLM model comparison currently expects RAG_LLM_PROVIDER=yandex")
    if not args.test_set.exists():
        build_test_set(output_path=args.test_set, size=args.max_questions)
    cases = load_jsonl(args.test_set)[: args.max_questions]
    if not cases:
        raise ValueError("Evaluation set is empty")

    retriever = HybridRetriever(settings)
    retriever.initialize()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrievals = {}
    for index, case in enumerate(cases, start=1):
        result = retriever.retrieve(
            case["question"],
            top_k=settings.top_k,
            context_window=settings.context_window,
            intro_chunks=settings.context_intro_chunks,
            max_context_chunks=settings.max_context_chunks,
        )
        retrievals[case["case_id"]] = result
        print(f"[{index}/{len(cases)}] retrieval: {case['question']}")

    rows: list[dict[str, Any]] = []
    for model in args.models:
        model_settings = replace(
            settings,
            yandex_model=model,
            yandex_fallback_models=(),
            generation_max_tokens=args.max_output_tokens,
        )
        generator = YandexGenerator(model_settings)
        print(f"Model: {model}")
        for index, case in enumerate(cases, start=1):
            retrieval = retrievals[case["case_id"]]
            row = {
                "case_id": case["case_id"],
                "question_id": case["question_id"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "gold_evidence": case["gold_evidence"],
                "required_terms": case.get("required_terms") or [],
                "gold_document_ids": case["gold_document_ids"],
                "retrieved_document_ids": [
                    chunk.document_id for chunk in retrieval.chunks
                ],
                "retrieved_chunks": [
                    {
                        "rank": chunk.rank,
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "score": chunk.score,
                    }
                    for chunk in retrieval.chunks
                ],
                "retrieval_hit_at_5": bool(
                    set(chunk.document_id for chunk in retrieval.chunks[:5])
                    & set(case["gold_document_ids"])
                ),
                "model_candidate": model,
                "prompt_strategy": PromptStrategy.STRUCTURED_OUTPUT.value,
                "temperature": settings.generation_temperature,
                "generation_error": None,
            }
            started = time.perf_counter()
            try:
                generation = generator.generate(
                    case["question"],
                    retrieval.chunks,
                    language="en",
                    strategy=PromptStrategy.STRUCTURED_OUTPUT,
                    temperature=settings.generation_temperature,
                    top_p=settings.generation_top_p,
                    max_output_tokens=args.max_output_tokens,
                )
                row.update(
                    {
                        "answer": generation.answer,
                        "raw_output": generation.raw_output,
                        "confidence": generation.confidence,
                        "source": generation.source,
                        "parse_success": generation.parse_success,
                        "schema_valid": generation.schema_valid,
                        "parse_error": generation.parse_error,
                        "generation_model": generation.model,
                        "generation_ms": round(generation.elapsed_ms, 3),
                        "input_tokens": generation.usage.input_tokens,
                        "output_tokens": generation.usage.output_tokens,
                        "total_tokens": generation.usage.total_tokens,
                    }
                )
                status = "ok"
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
                        "generation_model": model_settings.model_uri(model),
                        "generation_ms": round((time.perf_counter() - started) * 1000, 3),
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "generation_error": str(error),
                    }
                )
                status = "error"
            rows.append(row)
            print(f"  [{index}/{len(cases)}] {status}: {case['question']}")

    enrich_generation_rows(rows, retriever.encoder)
    summary = summarize(rows)
    best = select_best(summary)

    write_jsonl(args.output_dir / "llm_model_results.jsonl", rows)
    write_csv(args.output_dir / "llm_model_summary.csv", summary)
    write_json(args.output_dir / "llm_model_summary.json", summary)
    write_json(
        args.output_dir / "experiment_metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": list(args.models),
            "questions": len(cases),
            "prompt_strategy": PromptStrategy.STRUCTURED_OUTPUT.value,
            "best_model": best,
        },
    )
    args.report.write_text(build_report(summary, best, len(cases)), encoding="utf-8")
    print(f"Best model: {best['model_candidate']}")
    print(f"Report: {args.report}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_candidate"], []).append(row)
    output = []
    for model, group in grouped.items():
        successful = [row for row in group if not row.get("generation_error")]
        record = {
            "model_candidate": model,
            "runs": len(group),
            "successful_runs": len(successful),
            "generation_errors": len(group) - len(successful),
            "schema_adherence_rate": round(
                mean(row.get("schema_valid") is True for row in group),
                6,
            ),
            "citation_valid_rate": round(
                mean(row.get("citation_valid") for row in group),
                6,
            ),
            "answer_correct_rate": round(
                mean(row.get("answer_correct_heuristic") for row in group),
                6,
            ),
            "avg_semantic_similarity": round(
                mean(row.get("semantic_similarity") for row in group),
                6,
            ),
            "avg_token_f1": round(mean(row.get("token_f1") for row in group), 6),
            "avg_generation_ms": round(
                mean(row.get("generation_ms") for row in successful),
                3,
            ),
            "avg_total_tokens": round(
                mean(row.get("total_tokens") for row in successful),
                3,
            ),
        }
        record["selection_score"] = round(selection_score(record), 6)
        output.append(record)
    output.sort(
        key=lambda row: (
            row["selection_score"],
            -row["generation_errors"],
            -row["avg_generation_ms"],
        ),
        reverse=True,
    )
    return output


def selection_score(row: dict[str, Any]) -> float:
    return (
        0.25 * float(row["schema_adherence_rate"])
        + 0.30 * float(row["answer_correct_rate"])
        + 0.20 * float(row["avg_semantic_similarity"])
        + 0.15 * float(row["citation_valid_rate"])
        - 0.05 * min(float(row["avg_generation_ms"]) / 6000.0, 1.0)
        - 0.05 * min(float(row["generation_errors"]) / max(float(row["runs"]), 1.0), 1.0)
    )


def select_best(summary: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary:
        raise ValueError("No model summary rows")
    return summary[0]


def build_report(
    summary: list[dict[str, Any]],
    best: dict[str, Any],
    questions: int,
) -> str:
    lines = [
        "# Сравнение языковых моделей для RAG",
        "",
        f"Эксперимент выполнен на {questions} вопросах. Retrieval, чанки, prompt strategy и параметры генерации были одинаковыми для всех моделей. Менялась только LLM.",
        "",
        "Проверялись модели Yandex AI Studio, доступные через текущий Responses API.",
        "",
        "| Модель | Score | Correct | Semantic | Token F1 | Schema | Citations | Errors | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['model_candidate']}` | {row['selection_score']:.3f} | "
            f"{pct(row['answer_correct_rate'])} | {row['avg_semantic_similarity']:.3f} | "
            f"{row['avg_token_f1']:.3f} | {pct(row['schema_adherence_rate'])} | "
            f"{pct(row['citation_valid_rate'])} | {row['generation_errors']} | "
            f"{row['avg_generation_ms']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Выбранная модель",
            "",
            f"По сводной оценке выбрана `{best['model_candidate']}`.",
            "",
            "Score учитывает корректность ответа, семантическое сходство с эталоном/evidence, соблюдение structured output, валидность ссылок, ошибки генерации и задержку.",
            "",
            "Полные результаты сохранены в `data/experiments/llm_models/`.",
            "",
        ]
    )
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Yandex LLM models for the RAG pipeline.")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--max-output-tokens", type=int, default=700)
    args = parser.parse_args()
    if args.max_questions < 1:
        parser.error("--max-questions must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
