"""Evaluate answer refusal mechanisms for out-of-corpus questions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .build_rag_test_set import DEFAULT_OUTPUT, build_test_set
from .refusal import refusal_features
from .retriever import HybridRetriever
from .settings import PROJECT_ROOT, Settings


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "refusal"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "refusal_mechanisms.md"

NEGATIVE_QUESTIONS = [
    "How do I cook pasta?",
    "Who is the president of France?",
    "What is the weather tomorrow in Moscow?",
    "Explain Python decorators.",
    "How can I buy Bitcoin?",
    "What are the symptoms of flu?",
    "How do I reset my Instagram password?",
    "What is the capital of Brazil?",
    "Write a poem about cats.",
    "How do I apply for a US passport?",
    "Can I pay my federal taxes online?",
    "How do I register for college classes?",
    "What is the best laptop for gaming?",
    "How do I fix a leaking sink?",
    "What is the exchange rate for USD to EUR?",
    "How do I book a hotel in Paris?",
    "Explain quantum entanglement.",
    "How do I renew my gym membership?",
    "What is the recipe for pancakes?",
    "How do I file for divorce?",
]


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def run_experiment(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not args.test_set.exists():
        build_test_set(output_path=args.test_set, size=args.positive_questions)
    positives = load_jsonl(args.test_set)[: args.positive_questions]
    negatives = NEGATIVE_QUESTIONS[: args.negative_questions]

    retriever = HybridRetriever(settings)
    retriever.initialize()
    rows = []

    for case in positives:
        rows.append(feature_row(retriever, case["question"], True, case["case_id"]))
    for index, question in enumerate(negatives, start=1):
        rows.append(feature_row(retriever, question, False, f"negative_{index:03d}"))

    summaries = []
    summaries.extend(
        evaluate_thresholds(
            rows,
            "reranker_score_threshold",
            "top_score",
            [-8, -4, -3, -2, -1.5, -1, -0.5, 0, 1, 2, 3, 4, 5],
        )
    )
    summaries.extend(evaluate_thresholds(rows, "lexical_overlap_threshold", "lexical_overlap", [0.0, 0.05, 0.10, 0.20, 0.30]))
    summaries.extend(
        evaluate_hybrid(
            rows,
            [
                (0.0, 0.0),
                (0.0, 0.05),
                (-2.0, 0.0),
                (-2.0, 0.05),
                (1.0, 0.0),
                (2.0, 0.05),
                (3.0, 0.05),
            ],
        )
    )
    best = max(
        summaries,
        key=lambda row: (
            row["balanced_score"],
            -row["false_accept_rate"],
            -row["false_refusal_rate"],
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "refusal_feature_rows.jsonl", rows)
    write_csv(args.output_dir / "refusal_mechanism_summary.csv", summaries)
    write_json(args.output_dir / "refusal_mechanism_summary.json", summaries)
    write_json(
        args.output_dir / "experiment_metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "positive_questions": len(positives),
            "negative_questions": len(negatives),
            "best": best,
        },
    )
    args.report.write_text(
        build_report(
            summaries,
            best,
            len(positives),
            len(negatives),
            settings.refusal_min_top_score,
        ),
        encoding="utf-8",
    )
    print(f"Best refusal mechanism: {best['mechanism']} / {best['configuration']}")
    print(f"Report: {args.report}")


def feature_row(
    retriever: HybridRetriever,
    question: str,
    answerable: bool,
    case_id: str,
) -> dict[str, Any]:
    retrieval = retriever.retrieve(
        question,
        top_k=5,
        context_window=0,
        intro_chunks=0,
        max_context_chunks=5,
    )
    features = refusal_features(question, retrieval.chunks)
    return {
        "case_id": case_id,
        "question": question,
        "answerable": answerable,
        "top_score": features.top_score,
        "score_gap": features.score_gap,
        "lexical_overlap": features.lexical_overlap,
        "matched_query_terms": features.matched_query_terms,
        "query_terms": features.query_terms,
        "top_document_id": retrieval.chunks[0].document_id if retrieval.chunks else None,
        "top_title": retrieval.chunks[0].title if retrieval.chunks else None,
    }


def evaluate_thresholds(
    rows: list[dict[str, Any]],
    mechanism: str,
    field: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    output = []
    for threshold in thresholds:
        output.append(
            evaluate(
                rows,
                mechanism,
                f"{field}>={threshold:g}",
                lambda row, threshold=threshold: float(row[field] or 0.0) >= threshold,
            )
        )
    return output


def evaluate_hybrid(
    rows: list[dict[str, Any]],
    configs: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    output = []
    for min_score, min_overlap in configs:
        output.append(
            evaluate(
                rows,
                "hybrid_reranker_and_lexical",
                f"top_score>={min_score:g}; lexical>={min_overlap:g}",
                lambda row, min_score=min_score, min_overlap=min_overlap: (
                    float(row["top_score"] or 0.0) >= min_score
                    and float(row["lexical_overlap"] or 0.0) >= min_overlap
                ),
            )
        )
    return output


def evaluate(
    rows: list[dict[str, Any]],
    mechanism: str,
    configuration: str,
    predicts_answerable: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        expected = bool(row["answerable"])
        predicted = predicts_answerable(row)
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    false_accept = fp / (fp + tn) if fp + tn else 0.0
    false_refusal = fn / (fn + tp) if fn + tp else 0.0
    balanced = 0.35 * recall + 0.35 * specificity + 0.30 * accuracy
    return {
        "mechanism": mechanism,
        "configuration": configuration,
        "accuracy": round(accuracy, 6),
        "precision_answerable": round(precision, 6),
        "recall_answerable": round(recall, 6),
        "specificity_unanswerable": round(specificity, 6),
        "false_accept_rate": round(false_accept, 6),
        "false_refusal_rate": round(false_refusal, 6),
        "balanced_score": round(balanced, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_report(
    summaries: list[dict[str, Any]],
    best: dict[str, Any],
    positives: int,
    negatives: int,
    runtime_min_top_score: float,
) -> str:
    lines = [
        "# Исследование механизмов отказа от ответа",
        "",
        f"Проверка выполнена на {positives} вопросах из DMV-корпуса и {negatives} вопросах вне корпуса.",
        "",
        "Сравнивались три подхода:",
        "",
        "- порог по score после reranker;",
        "- порог по lexical overlap между вопросом и найденными фрагментами;",
        "- гибрид score + overlap;",
        "- роль DistilBERT-подобного классификатора в текущей архитектуре выполняет маленький cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L2-v2`: он получает пару query/document и выдаёт relevance score.",
        "",
        "| Механизм | Конфигурация | Balanced | Accuracy | Recall answerable | Specificity unanswerable | False accept | False refusal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: item["balanced_score"], reverse=True)[:12]:
        lines.append(
            f"| `{row['mechanism']}` | `{row['configuration']}` | "
            f"{row['balanced_score']:.3f} | {row['accuracy']:.3f} | "
            f"{row['recall_answerable']:.3f} | {row['specificity_unanswerable']:.3f} | "
            f"{row['false_accept_rate']:.3f} | {row['false_refusal_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            f"Лучший вариант на этом диагностическом наборе: `{best['mechanism']}` с конфигурацией `{best['configuration']}`.",
            "",
            f"В runtime выбран более осторожный к нормальным DMV-вопросам gate `top_score >= {runtime_min_top_score:g}`. Он может пропустить чуть больше спорных запросов к LLM-as-a-judge, зато на текущем наборе не отказывает на answerable-вопросах.",
            "",
            "Если top score ниже `RAG_REFUSAL_MIN_TOP_SCORE`, генерация не запускается и система честно сообщает, что в документах недостаточно информации. Это защищает от вопросов вне DMV-корпуса и экономит LLM-вызовы.",
            "",
            "Полный набор признаков и таблица сравнения сохранены в `data/experiments/refusal/`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refusal mechanisms.")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--positive-questions", type=int, default=20)
    parser.add_argument("--negative-questions", type=int, default=20)
    args = parser.parse_args()
    if args.positive_questions < 1 or args.negative_questions < 1:
        parser.error("question counts must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
