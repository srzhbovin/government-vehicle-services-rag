"""Token-aware routing experiment for a pool of LLM replicas."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


STRATEGIES = (
    "round_robin",
    "least_requests",
    "token_max",
    "token_ewma",
    "oracle",
)
CLASS_SPECS = {
    "faq": {"input": (180, 60), "output": (70, 25), "max_output": 512},
    "rag": {"input": (1100, 300), "output": (220, 90), "max_output": 700},
    "analysis": {"input": (3200, 800), "output": (850, 300), "max_output": 2048},
}
CLASS_WEIGHTS = ("faq", "faq", "faq", "rag", "rag", "rag", "analysis")


@dataclass(frozen=True)
class RequestProfile:
    request_id: str
    request_class: str
    input_tokens: int
    actual_output_tokens: int
    max_output_tokens: int


@dataclass
class ReplicaState:
    name: str
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    kv_capacity_tokens: int
    estimated_work_seconds: float = 0.0
    estimated_kv_tokens: int = 0
    initial_work_seconds: float = 0.0
    assignments: list[tuple[RequestProfile, int | None, float]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class AssignmentMeasurement:
    scenario: str
    strategy: str
    batch_size: int
    repeat: int
    request_id: str
    request_class: str
    replica: str
    input_tokens: int
    actual_output_tokens: int
    estimated_output_tokens: int | None
    max_output_tokens: int
    queue_ms: float
    service_ms: float
    end_to_end_ms: float


def generate_workload(batch_size: int, seed: int) -> list[RequestProfile]:
    rng = random.Random(seed)
    workload = []
    for index in range(batch_size):
        request_class = rng.choice(CLASS_WEIGHTS)
        spec = CLASS_SPECS[request_class]
        input_tokens = max(32, round(rng.gauss(*spec["input"])))
        output_tokens = max(
            8,
            min(
                int(spec["max_output"]),
                round(rng.gauss(*spec["output"])),
            ),
        )
        workload.append(
            RequestProfile(
                request_id=f"req-{index:04d}",
                request_class=request_class,
                input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                max_output_tokens=int(spec["max_output"]),
            )
        )
    rng.shuffle(workload)
    return workload


def calibrate_ewma(seed: int = 20260714, samples: int = 300, alpha: float = 0.15):
    rng = random.Random(seed)
    estimates = {
        request_class: float(spec["max_output"]) / 2
        for request_class, spec in CLASS_SPECS.items()
    }
    for _ in range(samples):
        request_class = rng.choice(tuple(CLASS_SPECS))
        spec = CLASS_SPECS[request_class]
        observed = max(
            8,
            min(int(spec["max_output"]), round(rng.gauss(*spec["output"]))),
        )
        estimates[request_class] = (
            alpha * observed + (1 - alpha) * estimates[request_class]
        )
    return {name: round(value) for name, value in estimates.items()}


def estimate_output_tokens(
    request: RequestProfile,
    strategy: str,
    ewma_estimates: dict[str, int],
) -> int | None:
    if strategy == "token_max":
        return request.max_output_tokens
    if strategy == "token_ewma":
        return ewma_estimates[request.request_class]
    if strategy == "oracle":
        return request.actual_output_tokens
    return None


def estimated_service_seconds(
    request: RequestProfile,
    output_tokens: int,
    replica: ReplicaState,
) -> float:
    return (
        request.input_tokens / replica.prefill_tokens_per_second
        + output_tokens / replica.decode_tokens_per_second
    )


def actual_service_seconds(request: RequestProfile, replica: ReplicaState) -> float:
    return estimated_service_seconds(request, request.actual_output_tokens, replica)


def route_workload(
    workload: Sequence[RequestProfile],
    strategy: str,
    *,
    replica_count: int,
    ewma_estimates: dict[str, int],
    prefill_tokens_per_second: float = 6000.0,
    decode_tokens_per_second: float = 300.0,
    kv_capacity_tokens: int = 32768,
    queue_penalty_seconds: float = 0.005,
    kv_pressure_weight_seconds: float = 0.25,
    initial_token_loads: Sequence[int] | None = None,
) -> list[ReplicaState]:
    initial_loads = list(initial_token_loads or [0] * replica_count)
    if len(initial_loads) != replica_count:
        raise ValueError("initial_token_loads must match replica_count")
    replicas = [
        ReplicaState(
            name=f"replica-{index + 1}",
            prefill_tokens_per_second=prefill_tokens_per_second,
            decode_tokens_per_second=decode_tokens_per_second,
            kv_capacity_tokens=kv_capacity_tokens,
            estimated_work_seconds=initial_loads[index] / decode_tokens_per_second,
            estimated_kv_tokens=initial_loads[index],
            initial_work_seconds=initial_loads[index] / decode_tokens_per_second,
        )
        for index in range(replica_count)
    ]
    round_robin_index = 0
    for request in workload:
        estimated_output = estimate_output_tokens(request, strategy, ewma_estimates)
        if strategy == "round_robin":
            selected = replicas[round_robin_index % len(replicas)]
            round_robin_index += 1
        elif strategy == "least_requests":
            selected = min(
                replicas, key=lambda item: (len(item.assignments), item.name)
            )
        else:
            assert estimated_output is not None

            def score(replica: ReplicaState) -> tuple[float, str]:
                request_work = estimated_service_seconds(
                    request, estimated_output, replica
                )
                kv_after_assignment = (
                    replica.estimated_kv_tokens
                    + request.input_tokens
                    + estimated_output
                )
                kv_pressure = kv_after_assignment / replica.kv_capacity_tokens
                value = (
                    replica.estimated_work_seconds
                    + request_work
                    + queue_penalty_seconds * len(replica.assignments)
                    + kv_pressure_weight_seconds * kv_pressure
                )
                return value, replica.name

            selected = min(replicas, key=score)
        routing_estimate = (
            estimated_output
            if estimated_output is not None
            else request.max_output_tokens
        )
        selected.assignments.append(
            (
                request,
                estimated_output,
                estimated_service_seconds(request, routing_estimate, selected),
            )
        )
        selected.estimated_work_seconds += estimated_service_seconds(
            request, routing_estimate, selected
        )
        selected.estimated_kv_tokens += request.input_tokens + routing_estimate
    return replicas


def measure_assignments(
    replicas: Sequence[ReplicaState],
    strategy: str,
    batch_size: int,
    repeat: int,
    scenario: str = "empty_pool",
) -> list[AssignmentMeasurement]:
    rows = []
    for replica in replicas:
        clock = replica.initial_work_seconds
        for request, estimated_output, _ in replica.assignments:
            service_seconds = actual_service_seconds(request, replica)
            queue_seconds = clock
            clock += service_seconds
            rows.append(
                AssignmentMeasurement(
                    scenario=scenario,
                    strategy=strategy,
                    batch_size=batch_size,
                    repeat=repeat,
                    request_id=request.request_id,
                    request_class=request.request_class,
                    replica=replica.name,
                    input_tokens=request.input_tokens,
                    actual_output_tokens=request.actual_output_tokens,
                    estimated_output_tokens=estimated_output,
                    max_output_tokens=request.max_output_tokens,
                    queue_ms=queue_seconds * 1000,
                    service_ms=service_seconds * 1000,
                    end_to_end_ms=clock * 1000,
                )
            )
    return rows


def summarize(rows: Sequence[AssignmentMeasurement]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[AssignmentMeasurement]] = defaultdict(list)
    for row in rows:
        grouped[(row.scenario, row.strategy, row.batch_size)].append(row)

    summary = []
    for (scenario, strategy, batch_size), items in sorted(grouped.items()):
        by_repeat: dict[int, list[AssignmentMeasurement]] = defaultdict(list)
        for item in items:
            by_repeat[item.repeat].append(item)
        run_metrics = []
        for repeat, repeat_items in sorted(by_repeat.items()):
            replica_work: dict[str, float] = defaultdict(float)
            for item in repeat_items:
                replica_work[item.replica] = max(
                    replica_work[item.replica], item.end_to_end_ms
                )
            makespan_ms = max(item.end_to_end_ms for item in repeat_items)
            actual_total_tokens = sum(
                item.input_tokens + item.actual_output_tokens for item in repeat_items
            )
            estimation_errors = [
                abs(item.estimated_output_tokens - item.actual_output_tokens)
                / item.actual_output_tokens
                for item in repeat_items
                if item.estimated_output_tokens is not None
            ]
            estimation_absolute_error = sum(
                abs(item.estimated_output_tokens - item.actual_output_tokens)
                for item in repeat_items
                if item.estimated_output_tokens is not None
            )
            estimation_actual_total = sum(
                item.actual_output_tokens
                for item in repeat_items
                if item.estimated_output_tokens is not None
            )
            run_metrics.append(
                {
                    "repeat": repeat,
                    "mean_e2e_ms": statistics.fmean(
                        item.end_to_end_ms for item in repeat_items
                    ),
                    "p95_e2e_ms": percentile(
                        (item.end_to_end_ms for item in repeat_items), 0.95
                    ),
                    "p95_queue_ms": percentile(
                        (item.queue_ms for item in repeat_items), 0.95
                    ),
                    "makespan_ms": makespan_ms,
                    "throughput_rps": len(repeat_items) / (makespan_ms / 1000),
                    "token_throughput_tps": actual_total_tokens / (makespan_ms / 1000),
                    "load_cv": coefficient_of_variation(replica_work.values()),
                    "load_max_min_ratio": max(replica_work.values())
                    / min(replica_work.values()),
                    "output_estimation_mape": (
                        statistics.fmean(estimation_errors)
                        if estimation_errors
                        else None
                    ),
                    "output_estimation_wape": (
                        estimation_absolute_error / estimation_actual_total
                        if estimation_actual_total
                        else None
                    ),
                }
            )
        summary.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "batch_size": batch_size,
                "repeats": len(run_metrics),
                **{
                    metric: round(
                        statistics.fmean(
                            run[metric]
                            for run in run_metrics
                            if run[metric] is not None
                        ),
                        4,
                    )
                    if any(run[metric] is not None for run in run_metrics)
                    else None
                    for metric in (
                        "mean_e2e_ms",
                        "p95_e2e_ms",
                        "p95_queue_ms",
                        "makespan_ms",
                        "throughput_rps",
                        "token_throughput_tps",
                        "load_cv",
                        "load_max_min_ratio",
                        "output_estimation_mape",
                        "output_estimation_wape",
                    )
                },
            }
        )
    return summary


def run_experiment(
    args: argparse.Namespace,
) -> tuple[list[AssignmentMeasurement], list[dict[str, object]], dict[str, object]]:
    ewma = calibrate_ewma(seed=args.seed)
    rows = []
    scenarios = {
        "empty_pool": [0] * args.replicas,
        "residual_load": [
            (15000, 4000, 9000)[index % 3] for index in range(args.replicas)
        ],
    }
    for scenario, initial_loads in scenarios.items():
        for batch_size in args.batch_sizes:
            for repeat in range(1, args.repeats + 1):
                workload = generate_workload(
                    batch_size, args.seed + batch_size * 100 + repeat
                )
                for strategy in STRATEGIES:
                    replicas = route_workload(
                        workload,
                        strategy,
                        replica_count=args.replicas,
                        ewma_estimates=ewma,
                        initial_token_loads=initial_loads,
                    )
                    rows.extend(
                        measure_assignments(
                            replicas,
                            strategy,
                            batch_size,
                            repeat,
                            scenario,
                        )
                    )
    metadata = {
        "status": "complete",
        "strategies": list(STRATEGIES),
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "replicas": args.replicas,
        "seed": args.seed,
        "prefill_tokens_per_second": 6000,
        "decode_tokens_per_second": 300,
        "kv_capacity_tokens": 32768,
        "queue_penalty_seconds": 0.005,
        "kv_pressure_weight_seconds": 0.25,
        "ewma_output_estimates": ewma,
        "scenarios": scenarios,
        "measured_assignments": len(rows),
    }
    return rows, summarize(rows), metadata


def write_results(
    output_dir: Path,
    rows: Sequence[AssignmentMeasurement],
    summary: Sequence[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "rows": summary}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    for scenario in metadata["scenarios"]:
        write_line_plot(
            plots / f"{scenario}_p95_latency_vs_batch.svg",
            summary,
            scenario=scenario,
            metric="p95_e2e_ms",
            title=f"p95 latency — {scenario}",
            y_label="milliseconds",
        )
        write_line_plot(
            plots / f"{scenario}_load_imbalance_vs_batch.svg",
            summary,
            scenario=scenario,
            metric="load_cv",
            title=f"Replica load imbalance — {scenario}",
            y_label="coefficient of variation",
        )
        write_line_plot(
            plots / f"{scenario}_throughput_vs_batch.svg",
            summary,
            scenario=scenario,
            metric="throughput_rps",
            title=f"Request throughput — {scenario}",
            y_label="requests / second",
        )
    (output_dir / "report.md").write_text(
        build_report(summary, metadata), encoding="utf-8", newline="\n"
    )


def build_report(
    summary: Sequence[dict[str, object]], metadata: dict[str, object]
) -> str:
    indexed = {
        (str(row["scenario"]), str(row["strategy"]), int(row["batch_size"])): row
        for row in summary
    }
    lines = [
        "# Token-aware routing между LLM-репликами",
        "",
        "Статус: `complete`.",
        "",
        f"Сравнено {len(metadata['strategies'])} стратегий на {metadata['replicas']} репликах, "
        f"размеры burst: {', '.join(map(str, metadata['batch_sizes']))}, повторов: {metadata['repeats']}.",
        "",
        "В `residual_load` реплики начинают с 15 000, 4 000 и 9 000 незавершённых токенов соответственно.",
    ]
    for scenario in metadata["scenarios"]:
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                "| Batch | Strategy | p95 E2E, ms | Makespan, ms | Throughput, req/s | Load CV | Output WAPE |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for batch_size in metadata["batch_sizes"]:
            for strategy in STRATEGIES:
                row = indexed[(scenario, strategy, int(batch_size))]
                wape = row["output_estimation_wape"]
                lines.append(
                    f"| {batch_size} | {strategy} | {row['p95_e2e_ms']:.1f} | "
                    f"{row['makespan_ms']:.1f} | {row['throughput_rps']:.2f} | "
                    f"{row['load_cv']:.3f} | "
                    f"{'—' if wape is None else f'{float(wape) * 100:.1f}%'} |"
                )
    largest = int(max(metadata["batch_sizes"]))
    rr = indexed[("residual_load", "round_robin", largest)]
    ewma = indexed[("residual_load", "token_ewma", largest)]
    p95_gain = (1 - float(ewma["p95_e2e_ms"]) / float(rr["p95_e2e_ms"])) * 100
    throughput_gain = (
        float(ewma["throughput_rps"]) / float(rr["throughput_rps"]) - 1
    ) * 100
    lines.extend(
        [
            "",
            "## Итог",
            "",
            f"В residual_load при burst={largest} EWMA token-aware снизил p95 latency на {p95_gain:.1f}% и изменил throughput на {throughput_gain:+.1f}% относительно Round Robin. Oracle используется как эталон маршрутизации с точной оценкой стоимости; даже он остаётся жадным алгоритмом и не обязан минимизировать каждую метрику одновременно.",
            "",
            f"EWMA-калибровка: `{json.dumps(metadata['ewma_output_estimates'], sort_keys=True)}`.",
            "",
            "Графики находятся в `plots/`. Эксперимент является детерминированной моделью маршрутизации и не заменяет повторный прогон на реальных LiteLLM-репликах.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_line_plot(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    scenario: str,
    metric: str,
    title: str,
    y_label: str,
) -> None:
    rows = [row for row in rows if row["scenario"] == scenario]
    width, height = 920, 520
    left, right, top, bottom = 90, 30, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    batches = sorted({int(row["batch_size"]) for row in rows})
    values = [float(row[metric]) for row in rows if row[metric] is not None]
    y_max = max(values) * 1.08 if values else 1.0
    colors = {
        "round_robin": "#64748b",
        "least_requests": "#f59e0b",
        "token_max": "#8b5cf6",
        "token_ewma": "#059669",
        "oracle": "#2563eb",
    }

    def x_position(batch: int) -> float:
        index = batches.index(batch)
        return left + index * plot_width / max(1, len(batches) - 1)

    def y_position(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334155"/>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>',
            ]
        )
    for batch in batches:
        x = x_position(batch)
        svg.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{batch}</text>'
        )
    indexed = {(str(row["strategy"]), int(row["batch_size"])): row for row in rows}
    for legend_index, strategy in enumerate(STRATEGIES):
        points = " ".join(
            f"{x_position(batch):.1f},{y_position(float(indexed[(strategy, batch)][metric])):.1f}"
            for batch in batches
        )
        color = colors[strategy]
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for batch in batches:
            row = indexed[(strategy, batch)]
            svg.append(
                f'<circle cx="{x_position(batch):.1f}" cy="{y_position(float(row[metric])):.1f}" r="4" fill="{color}"/>'
            )
        legend_x = left + legend_index * 155
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 25}" x2="{legend_x + 24}" y2="{height - 25}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{height - 21}" font-family="sans-serif" font-size="11">{strategy}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 48}" text-anchor="middle" font-family="sans-serif" font-size="12">simultaneous requests</text>',
            f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">{y_label}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg), encoding="utf-8", newline="\n")


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def coefficient_of_variation(values: Iterable[float]) -> float:
    data = list(values)
    mean = statistics.fmean(data)
    return statistics.pstdev(data) / mean if mean else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[12, 36, 72, 144])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/priority_gateway/token_aware"),
    )
    args = parser.parse_args(argv)
    if (
        args.repeats < 1
        or args.replicas < 2
        or any(size < 1 for size in args.batch_sizes)
    ):
        parser.error(
            "positive batch sizes/repeats and at least two replicas are required"
        )
    return args


def main() -> None:
    args = parse_args()
    rows, summary, metadata = run_experiment(args)
    write_results(args.output_dir, rows, summary, metadata)


if __name__ == "__main__":
    main()
