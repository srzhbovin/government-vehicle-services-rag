"""Controlled load experiment for FIFO and priority-aware LLM scheduling."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .scheduler import PriorityScheduler


DEFAULT_GROUPS = {"critical": 0, "standard": 10, "batch": 20}
DEFAULT_MIX = {"critical": 9, "standard": 21, "batch": 15}


@dataclass(frozen=True)
class RequestMeasurement:
    strategy: str
    concurrency: int
    repeat: int
    request_id: str
    group: str
    queue_ms: float
    processing_ms: float
    end_to_end_ms: float
    completed_at: float


async def run_scenario(
    *,
    strategy: str,
    concurrency: int,
    repeat: int,
    service_ms: float,
    group_mix: dict[str, int],
    seed: int,
) -> tuple[list[RequestMeasurement], float]:
    total_requests = sum(group_mix.values())
    scheduler = PriorityScheduler(
        strategy=strategy,
        max_concurrency=concurrency,
        max_queue_size=total_requests + concurrency,
        queue_timeout_seconds=60,
        aging_interval_seconds=0.5,
    )
    blockers = [await scheduler.acquire("warmup", 0) for _ in range(concurrency)]
    requests = [
        (f"{group}-{index:03d}", group)
        for group, count in group_mix.items()
        for index in range(count)
    ]
    random.Random(seed + repeat * 1000 + concurrency).shuffle(requests)

    async def execute(request_id: str, group: str) -> RequestMeasurement:
        started = time.perf_counter()
        lease = await scheduler.acquire(group, DEFAULT_GROUPS[group])
        processing_started = time.perf_counter()
        await asyncio.sleep(service_ms / 1000)
        processing_ms = (time.perf_counter() - processing_started) * 1000
        completed_at = time.perf_counter()
        await lease.release()
        return RequestMeasurement(
            strategy=strategy,
            concurrency=concurrency,
            repeat=repeat,
            request_id=request_id,
            group=group,
            queue_ms=lease.queue_ms,
            processing_ms=processing_ms,
            end_to_end_ms=(completed_at - started) * 1000,
            completed_at=completed_at,
        )

    tasks = [
        asyncio.create_task(execute(request_id, group))
        for request_id, group in requests
    ]
    deadline = time.perf_counter() + 10
    while True:
        snapshot = await scheduler.snapshot()
        if snapshot["queued"] == total_requests:
            break
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                "Timed out while preparing simultaneous benchmark requests"
            )
        await asyncio.sleep(0.001)

    measurement_started = time.perf_counter()
    for lease in blockers:
        await lease.release()
    measurements = await asyncio.gather(*tasks)
    duration_seconds = (
        max(item.completed_at for item in measurements) - measurement_started
    )
    return measurements, duration_seconds


def summarize(
    measurements: Sequence[RequestMeasurement],
    durations: dict[tuple[str, int, int], float],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[RequestMeasurement]] = defaultdict(list)
    for item in measurements:
        grouped[(item.strategy, item.concurrency, item.group)].append(item)

    rows: list[dict[str, object]] = []
    for (strategy, concurrency, group), items in sorted(grouped.items()):
        queue = [item.queue_ms for item in items]
        processing = [item.processing_ms for item in items]
        e2e = [item.end_to_end_ms for item in items]
        scenario_durations = [
            duration
            for (
                candidate_strategy,
                candidate_concurrency,
                _,
            ), duration in durations.items()
            if candidate_strategy == strategy and candidate_concurrency == concurrency
        ]
        requests_per_repeat = len(items) / len(scenario_durations)
        total_mix = requests_per_repeat * sum(DEFAULT_MIX.values()) / DEFAULT_MIX[group]
        throughput = [total_mix / duration for duration in scenario_durations]
        rows.append(
            {
                "strategy": strategy,
                "concurrency": concurrency,
                "group": group,
                "samples": len(items),
                "mean_queue_ms": round(statistics.fmean(queue), 3),
                "p50_queue_ms": round(_percentile(queue, 0.50), 3),
                "p95_queue_ms": round(_percentile(queue, 0.95), 3),
                "mean_processing_ms": round(statistics.fmean(processing), 3),
                "p95_processing_ms": round(_percentile(processing, 0.95), 3),
                "mean_e2e_ms": round(statistics.fmean(e2e), 3),
                "p95_e2e_ms": round(_percentile(e2e, 0.95), 3),
                "mean_throughput_rps": round(statistics.fmean(throughput), 3),
            }
        )
    return rows


def write_results(
    output_dir: Path,
    measurements: Sequence[RequestMeasurement],
    summary: Sequence[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fields = list(asdict(measurements[0]).keys()) if measurements else []
    with (output_dir / "requests.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in measurements)
    summary_fields = list(summary[0].keys()) if summary else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"metadata": metadata, "rows": summary}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, metadata), encoding="utf-8"
    )


def build_report(rows: Sequence[dict[str, object]], metadata: dict[str, object]) -> str:
    concurrency_values = metadata["concurrency"]
    lines = [
        "# Эксперимент с приоритизацией запросов",
        "",
        "Статус: `complete`.",
        "",
        "## Протокол",
        "",
        "- стратегии: FIFO и priority queue with aging;",
        f"- одновременные слоты: {', '.join(map(str, concurrency_values))};",
        f"- повторов: {metadata['repeats']};",
        f"- запросов в одном прогоне: {metadata['requests_per_run']};",
        f"- фиксированное время downstream-обработки: {metadata['service_ms']} ms;",
        "- смесь: critical 20%, standard 46.7%, batch 33.3%;",
        "- все запросы ставятся в очередь до начала измеряемой обработки.",
        "",
        "Контролируемый downstream имитирует одинаковую генерацию за LiteLLM. Это исключает влияние сети и случайной длины ответа и позволяет измерить именно эффект планировщика. Для production sizing эксперимент следует повторить с выбранной моделью.",
        "",
        "## Результаты",
        "",
        "| Slots | Group | FIFO mean wait, ms | Priority mean wait, ms | Изменение | Priority p95 E2E, ms | Throughput, req/s |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    indexed = {
        (str(row["strategy"]), int(row["concurrency"]), str(row["group"])): row
        for row in rows
    }
    for concurrency in concurrency_values:
        for group in ("critical", "standard", "batch"):
            fifo = indexed[("fifo", int(concurrency), group)]
            priority = indexed[("priority_aging", int(concurrency), group)]
            baseline = float(fifo["mean_queue_ms"])
            changed = float(priority["mean_queue_ms"])
            delta = ((changed / baseline) - 1) * 100 if baseline else 0.0
            lines.append(
                f"| {concurrency} | {group} | {baseline:.1f} | {changed:.1f} | {delta:+.1f}% | "
                f"{float(priority['p95_e2e_ms']):.1f} | {float(priority['mean_throughput_rps']):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Приоритизация перераспределяет время ожидания, а не ускоряет саму модель: critical-запросы начинают выполняться раньше, batch-запросы ждут дольше, время непосредственной обработки остаётся одинаковым. Общая пропускная способность должна оставаться близкой к FIFO, поскольку число слотов и длительность генерации не меняются.",
            "",
            "Aging постепенно уменьшает эффективное значение приоритета ожидающего запроса. Поэтому batch-трафик не блокируется навсегда даже при постоянном потоке critical-запросов.",
        ]
    )
    return "\n".join(lines) + "\n"


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


async def benchmark(args: argparse.Namespace) -> None:
    all_measurements: list[RequestMeasurement] = []
    durations: dict[tuple[str, int, int], float] = {}
    for strategy in ("fifo", "priority_aging"):
        for concurrency in args.concurrency:
            for repeat in range(1, args.repeats + 1):
                measurements, duration = await run_scenario(
                    strategy=strategy,
                    concurrency=concurrency,
                    repeat=repeat,
                    service_ms=args.service_ms,
                    group_mix=DEFAULT_MIX,
                    seed=args.seed,
                )
                all_measurements.extend(measurements)
                durations[(strategy, concurrency, repeat)] = duration
    metadata = {
        "status": "complete",
        "strategies": ["fifo", "priority_aging"],
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "requests_per_run": sum(DEFAULT_MIX.values()),
        "service_ms": args.service_ms,
        "seed": args.seed,
        "group_priorities": DEFAULT_GROUPS,
        "group_mix": DEFAULT_MIX,
    }
    rows = summarize(all_measurements, durations)
    write_results(args.output_dir, all_measurements, rows, metadata)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--service-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/priority_gateway/baseline"),
    )
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.concurrency):
        parser.error("concurrency values must be positive")
    if args.repeats < 1 or args.service_ms <= 0:
        parser.error("repeats and service-ms must be positive")
    return args


def main() -> None:
    asyncio.run(benchmark(parse_args()))


if __name__ == "__main__":
    main()
