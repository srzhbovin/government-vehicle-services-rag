"""Aggregate reproducible SGLang, vLLM, and LMDeploy benchmarks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "data" / "experiments" / "inference_engines"
DEFAULT_ENGINES = ("sglang", "vllm", "lmdeploy")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16)
DEFAULT_REPEATS = 3
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42

RAW_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "duration_s": ("duration", "duration_s"),
    "request_throughput": ("request_throughput", "request_throughput_rps"),
    "input_throughput": ("input_throughput", "input_token_throughput"),
    "output_throughput": ("output_throughput", "output_token_throughput"),
    "total_throughput": ("total_throughput", "total_token_throughput"),
    "mean_e2e_latency_ms": ("mean_e2e_latency_ms", "avg_e2e_latency_ms"),
    "median_e2e_latency_ms": ("median_e2e_latency_ms",),
    "p95_e2e_latency_ms": ("p95_e2e_latency_ms", "e2e_latency_p95_ms"),
    "p99_e2e_latency_ms": ("p99_e2e_latency_ms", "e2e_latency_p99_ms"),
    "mean_ttft_ms": ("mean_ttft_ms", "avg_ttft_ms"),
    "median_ttft_ms": ("median_ttft_ms",),
    "p95_ttft_ms": ("p95_ttft_ms", "ttft_p95_ms"),
    "p99_ttft_ms": ("p99_ttft_ms", "ttft_p99_ms"),
    "mean_tpot_ms": ("mean_tpot_ms", "avg_tpot_ms"),
    "median_tpot_ms": ("median_tpot_ms",),
    "p95_tpot_ms": ("p95_tpot_ms", "tpot_p95_ms"),
    "p99_tpot_ms": ("p99_tpot_ms", "tpot_p99_ms"),
    "mean_itl_ms": ("mean_itl_ms", "avg_itl_ms"),
    "median_itl_ms": ("median_itl_ms",),
    "p95_itl_ms": ("p95_itl_ms", "itl_p95_ms"),
    "p99_itl_ms": ("p99_itl_ms", "itl_p99_ms"),
    "concurrency": ("concurrency",),
}

TELEMETRY_METRICS = ("peak_vram_mib", "avg_gpu_util_pct", "avg_power_w")
REQUIRED_SUCCESS_METRICS = (
    "duration_s",
    "request_throughput",
    "output_throughput",
    "p95_ttft_ms",
    "p95_tpot_ms",
    "p95_e2e_latency_ms",
)
SENSITIVE_NAME_PARTS = ("KEY", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True, order=True)
class RunKey:
    engine: str
    batch_size: int
    repeat: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "batch_size": self.batch_size,
            "repeat": self.repeat,
        }

    @property
    def label(self) -> str:
        return f"{self.engine}|batch={self.batch_size}|repeat={self.repeat}"


@dataclass(frozen=True)
class BenchmarkRun:
    key: RunKey
    source: str
    metrics: dict[str, float | None]
    requested_requests: int | None
    completed_requests: int | None
    failed_requests: int | None
    error_rate: float | None


@dataclass(frozen=True)
class TelemetryRun:
    key: RunKey
    source: str
    samples: int
    peak_vram_mib: float | None
    avg_gpu_util_pct: float | None
    avg_power_w: float | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "na", "none", "null", "[n/a]"}:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def normalize_engine(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-")
    if text.startswith("sglang"):
        return "sglang"
    if text.startswith("vllm"):
        return "vllm"
    if text.startswith("lmdeploy"):
        return "lmdeploy"
    return text or None


def parse_string_sequence(value: str) -> tuple[str, ...]:
    items = tuple(item for item in re.split(r"[,;\s]+", value.strip()) if item)
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty list")
    return items


def parse_int_sequence(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(
            int(item) for item in re.split(r"[,;\s]+", value.strip()) if item
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a list of integers") from error
    if not numbers or any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return numbers


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    confidence: float = 0.95,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return None, None
    if array.size == 1 or samples <= 1:
        value = float(statistic(array))
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    sampled = array[indices]
    estimates = np.apply_along_axis(statistic, 1, sampled)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def describe_values(
    values: Iterable[float | None],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int | None]:
    clean = [
        float(value) for value in values if value is not None and math.isfinite(value)
    ]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    array = np.asarray(clean, dtype=float)
    low, high = bootstrap_confidence_interval(
        clean,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
    }


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = "<redacted>" if is_sensitive_name(key) else value
    return values


def is_sensitive_name(name: str) -> bool:
    upper = name.upper()
    if any(part in upper for part in SENSITIVE_NAME_PARTS):
        return True
    return "TOKEN" in upper and not upper.endswith("_TOKENS")


def first_value(record: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def infer_number(text: str, labels: Sequence[str]) -> int | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = rf"(?:^|[^a-z0-9])(?:{label_pattern})(?:[_=:.-]*)(\d+)(?:[^0-9]|$)"
    match = re.search(pattern, text.lower())
    return int(match.group(1)) if match else None


def infer_engine(record: dict[str, Any], path: Path) -> str | None:
    direct = normalize_engine(first_value(record, ("engine", "backend")))
    if direct:
        return direct
    tag = str(record.get("tag") or "")
    for candidate in DEFAULT_ENGINES:
        if re.search(rf"(?:^|[^a-z]){candidate}(?:[^a-z]|$)", tag.lower()):
            return candidate
    path_text = path.as_posix().lower()
    for candidate in DEFAULT_ENGINES:
        if re.search(rf"(?:^|[^a-z]){candidate}(?:[^a-z]|$)", path_text):
            return candidate
    return None


def infer_batch_size(record: dict[str, Any], path: Path) -> int | None:
    direct = safe_int(first_value(record, ("batch_size", "max_concurrency")))
    if direct is not None:
        return direct
    labels = ("batch", "batch-size", "batch_size", "bs", "b")
    for text in (f"{record.get('tag') or ''} {path.name}", path.as_posix()):
        inferred = infer_number(text, labels)
        if inferred is not None:
            return inferred
    return None


def infer_repeat(record: dict[str, Any], path: Path) -> int | None:
    direct = safe_int(first_value(record, ("repeat", "repetition", "run_index")))
    if direct is not None:
        return direct
    labels = ("repeat", "repetition", "rep", "run", "r")
    for text in (f"{record.get('tag') or ''} {path.name}", path.as_posix()):
        inferred = infer_number(text, labels)
        if inferred is not None:
            return inferred
    return None


def percentile_from_seconds(values: Any, percentile: float) -> float | None:
    if not isinstance(values, list):
        return None
    clean = [safe_float(value) for value in values]
    array = np.asarray([value for value in clean if value is not None], dtype=float)
    if array.size == 0:
        return None
    return float(np.percentile(array, percentile) * 1000.0)


def extract_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    metrics = {
        name: safe_float(first_value(record, aliases))
        for name, aliases in RAW_METRIC_ALIASES.items()
    }
    detail_fallbacks = {
        "p95_ttft_ms": ("ttfts",),
        "p95_e2e_latency_ms": ("e2e_latencies", "latencies"),
        "p95_tpot_ms": ("tpots",),
    }
    for metric, aliases in detail_fallbacks.items():
        if metrics[metric] is not None:
            continue
        for alias in aliases:
            value = percentile_from_seconds(record.get(alias), 95.0)
            if value is not None:
                metrics[metric] = value
                break
    return metrics


def extract_request_counts(
    record: dict[str, Any],
    expected_requests: int | None,
) -> tuple[int | None, int | None, int | None, float | None]:
    completed = safe_int(
        first_value(record, ("completed", "completed_requests", "successful_requests"))
    )
    requested = safe_int(
        first_value(record, ("num_prompts", "requested", "total_requests"))
    )
    errors = record.get("errors")
    failed: int | None = None
    if isinstance(errors, list):
        requested = len(errors)
        failed = sum(bool(str(error).strip()) for error in errors if error is not None)
    if requested is None and isinstance(record.get("input_lens"), list):
        requested = len(record["input_lens"])
    if requested is None:
        requested = expected_requests
    if completed is None and requested is not None and failed is not None:
        completed = max(0, requested - failed)
    if failed is None and requested is not None and completed is not None:
        failed = max(0, requested - completed)
    error_rate = (
        failed / requested
        if failed is not None and requested is not None and requested > 0
        else None
    )
    return requested, completed, failed, error_rate


def load_benchmark_runs(
    raw_dir: Path,
    *,
    expected_requests: int | None = None,
) -> tuple[list[BenchmarkRun], list[str]]:
    runs: list[BenchmarkRun] = []
    issues: list[str] = []
    if not raw_dir.exists():
        return runs, [f"Raw directory does not exist: {raw_dir}"]
    for path in sorted(raw_dir.rglob("*.jsonl")):
        fallback_repeats: Counter[tuple[str, int]] = Counter()
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as error:
            issues.append(f"Cannot read {path}: {error}")
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(f"Invalid JSON in {path}:{line_number}: {error.msg}")
                continue
            if not isinstance(record, dict):
                issues.append(f"Expected a JSON object in {path}:{line_number}")
                continue
            engine = infer_engine(record, path)
            batch_size = infer_batch_size(record, path)
            if not engine or batch_size is None or batch_size <= 0:
                issues.append(f"Cannot infer engine/batch in {path}:{line_number}")
                continue
            fallback_repeats[(engine, batch_size)] += 1
            repeat = (
                infer_repeat(record, path) or fallback_repeats[(engine, batch_size)]
            )
            if repeat <= 0:
                issues.append(f"Invalid repeat in {path}:{line_number}")
                continue
            requested, completed, failed, error_rate = extract_request_counts(
                record,
                expected_requests,
            )
            runs.append(
                BenchmarkRun(
                    key=RunKey(engine, batch_size, repeat),
                    source=f"{path}:{line_number}",
                    metrics=extract_metrics(record),
                    requested_requests=requested,
                    completed_requests=completed,
                    failed_requests=failed,
                    error_rate=error_rate,
                )
            )
    return runs, issues


def normalize_header(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {
        normalize_header(key): value for key, value in row.items() if key is not None
    }


def row_value(row: dict[str, str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        if alias in row:
            return row[alias]
    return None


def telemetry_identity(row: dict[str, str], path: Path) -> RunKey | None:
    engine = normalize_engine(row_value(row, ("engine", "backend")))
    if not engine:
        engine = infer_engine({}, path)
    batch = safe_int(row_value(row, ("batch_size", "batch", "max_concurrency")))
    if batch is None:
        batch = infer_batch_size({}, path)
    repeat = safe_int(row_value(row, ("repeat", "repetition", "run", "run_index")))
    if repeat is None:
        repeat = infer_repeat({}, path)
    if not engine or batch is None or repeat is None or batch <= 0 or repeat <= 0:
        return None
    return RunKey(engine, batch, repeat)


def load_telemetry_runs(telemetry_dir: Path) -> tuple[list[TelemetryRun], list[str]]:
    telemetry: list[TelemetryRun] = []
    issues: list[str] = []
    if not telemetry_dir.exists():
        return telemetry, [f"Telemetry directory does not exist: {telemetry_dir}"]
    for path in sorted(telemetry_dir.rglob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as input_file:
                rows = [normalized_row(row) for row in csv.DictReader(input_file)]
        except (OSError, csv.Error) as error:
            issues.append(f"Cannot read telemetry {path}: {error}")
            continue
        if not rows:
            issues.append(f"Telemetry file is empty: {path}")
            continue
        grouped: dict[RunKey, list[dict[str, str]]] = defaultdict(list)
        for row_number, row in enumerate(rows, start=2):
            key = telemetry_identity(row, path)
            if key is None:
                issues.append(f"Cannot infer telemetry identity in {path}:{row_number}")
                continue
            grouped[key].append(row)
        for key, samples in grouped.items():
            memories = [
                safe_float(
                    row_value(
                        row, ("memory_used_mib", "used_gpu_memory_mib", "vram_mib")
                    )
                )
                for row in samples
            ]
            utilizations = [
                safe_float(
                    row_value(
                        row,
                        (
                            "utilization_gpu",
                            "gpu_utilization_percent",
                            "gpu_util_pct",
                            "gpu_util",
                        ),
                    )
                )
                for row in samples
            ]
            powers = [
                safe_float(row_value(row, ("power_draw_w", "power_w", "power_draw")))
                for row in samples
            ]
            clean_memory = [value for value in memories if value is not None]
            clean_util = [value for value in utilizations if value is not None]
            clean_power = [value for value in powers if value is not None]
            if not clean_memory and not clean_util and not clean_power:
                issues.append(f"No recognized telemetry columns in {path}")
                continue
            telemetry.append(
                TelemetryRun(
                    key=key,
                    source=str(path),
                    samples=len(samples),
                    peak_vram_mib=max(clean_memory) if clean_memory else None,
                    avg_gpu_util_pct=float(np.mean(clean_util)) if clean_util else None,
                    avg_power_w=float(np.mean(clean_power)) if clean_power else None,
                )
            )
    return telemetry, issues


def expected_run_keys(
    engines: Sequence[str],
    batch_sizes: Sequence[int],
    repeats: int,
) -> set[RunKey]:
    return {
        RunKey(engine, batch_size, repeat)
        for engine in engines
        for batch_size in batch_sizes
        for repeat in range(1, repeats + 1)
    }


def count_details(
    counts: Counter[RunKey], predicate: Callable[[int], bool]
) -> list[dict[str, Any]]:
    return [
        {**key.as_dict(), "count": count}
        for key, count in sorted(counts.items())
        if predicate(count)
    ]


def validate_run_data(
    run: BenchmarkRun,
    *,
    expected_requests: int | None,
) -> list[str]:
    issues: list[str] = []
    requested = run.requested_requests
    completed = run.completed_requests
    failed = run.failed_requests

    if (
        requested is None
        or completed is None
        or failed is None
        or run.error_rate is None
    ):
        issues.append("request/completion/error counts are incomplete")
    else:
        if expected_requests is not None and requested != expected_requests:
            issues.append(f"expected {expected_requests} requests, got {requested}")
        if completed + failed != requested:
            issues.append(
                f"completed ({completed}) + failed ({failed}) does not equal requested ({requested})"
            )
        expected_error_rate = failed / requested if requested else None
        if expected_error_rate is None or not math.isclose(
            run.error_rate,
            expected_error_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            issues.append("error rate is inconsistent with request counts")

    if completed is not None and completed > 0:
        missing = [
            name for name in REQUIRED_SUCCESS_METRICS if run.metrics.get(name) is None
        ]
        if missing:
            issues.append("missing required metrics: " + ", ".join(missing))
        for name in ("duration_s", "request_throughput", "output_throughput"):
            value = run.metrics.get(name)
            if value is not None and value <= 0:
                issues.append(f"{name} must be positive when requests completed")
        for name in ("p95_ttft_ms", "p95_tpot_ms", "p95_e2e_latency_ms"):
            value = run.metrics.get(name)
            if value is not None and value < 0:
                issues.append(f"{name} cannot be negative")

    return issues


def validate_completeness(
    runs: Sequence[BenchmarkRun],
    telemetry: Sequence[TelemetryRun],
    *,
    engines: Sequence[str],
    batch_sizes: Sequence[int],
    repeats: int,
    raw_issues: Sequence[str] = (),
    telemetry_issues: Sequence[str] = (),
    require_telemetry: bool = True,
    expected_requests: int | None = None,
) -> dict[str, Any]:
    expected = expected_run_keys(engines, batch_sizes, repeats)
    raw_counts = Counter(run.key for run in runs)
    telemetry_counts = Counter(item.key for item in telemetry)
    missing_raw = sorted(expected - set(raw_counts))
    unexpected_raw = sorted(set(raw_counts) - expected)
    missing_telemetry = sorted(expected - set(telemetry_counts))
    unexpected_telemetry = sorted(set(telemetry_counts) - expected)
    duplicate_raw = count_details(raw_counts, lambda count: count > 1)
    duplicate_telemetry = count_details(telemetry_counts, lambda count: count > 1)
    invalid_run_data = [
        {**run.key.as_dict(), "issues": issues}
        for run in runs
        if (issues := validate_run_data(run, expected_requests=expected_requests))
    ]

    if not runs:
        status = "not_run"
    else:
        raw_incomplete = bool(
            missing_raw
            or unexpected_raw
            or duplicate_raw
            or raw_issues
            or invalid_run_data
        )
        telemetry_incomplete = require_telemetry and bool(
            missing_telemetry
            or unexpected_telemetry
            or duplicate_telemetry
            or telemetry_issues
        )
        status = "incomplete" if raw_incomplete or telemetry_incomplete else "complete"

    return {
        "status": status,
        "expected_runs": len(expected),
        "raw_records": len(runs),
        "unique_raw_runs": len(raw_counts),
        "telemetry_records": len(telemetry),
        "unique_telemetry_runs": len(telemetry_counts),
        "missing_raw_runs": [key.as_dict() for key in missing_raw],
        "duplicate_raw_runs": duplicate_raw,
        "unexpected_raw_runs": [key.as_dict() for key in unexpected_raw],
        "missing_telemetry_runs": [key.as_dict() for key in missing_telemetry],
        "duplicate_telemetry_runs": duplicate_telemetry,
        "unexpected_telemetry_runs": [key.as_dict() for key in unexpected_telemetry],
        "invalid_run_data": invalid_run_data,
        "raw_issues": list(raw_issues),
        "telemetry_issues": list(telemetry_issues),
        "telemetry_required": require_telemetry,
    }


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return (base_seed + zlib.crc32(payload)) % (2**32)


def add_statistics(
    row: dict[str, Any],
    metric: str,
    values: Iterable[float | None],
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    stats = describe_values(values, bootstrap_samples=bootstrap_samples, seed=seed)
    for name, value in stats.items():
        row[f"{metric}_{name}"] = value


def summarize_runs(
    runs: Sequence[BenchmarkRun],
    telemetry: Sequence[TelemetryRun],
    *,
    expected_repeats: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    grouped_runs: dict[tuple[str, int], list[BenchmarkRun]] = defaultdict(list)
    grouped_telemetry: dict[tuple[str, int], list[TelemetryRun]] = defaultdict(list)
    for run in runs:
        grouped_runs[(run.key.engine, run.key.batch_size)].append(run)
    for item in telemetry:
        grouped_telemetry[(item.key.engine, item.key.batch_size)].append(item)

    rows: list[dict[str, Any]] = []
    for engine, batch_size in sorted(grouped_runs):
        items = grouped_runs[(engine, batch_size)]
        telemetry_items = grouped_telemetry.get((engine, batch_size), [])
        row: dict[str, Any] = {
            "engine": engine,
            "batch_size": batch_size,
            "expected_repeats": expected_repeats,
            "observed_runs": len(items),
            "unique_repeats": len({item.key.repeat for item in items}),
            "telemetry_runs": len(telemetry_items),
            "telemetry_samples": sum(item.samples for item in telemetry_items),
        }
        for metric in RAW_METRIC_ALIASES:
            add_statistics(
                row,
                metric,
                (item.metrics.get(metric) for item in items),
                bootstrap_samples=bootstrap_samples,
                seed=stable_seed(bootstrap_seed, engine, batch_size, metric),
            )
        telemetry_values = {
            "peak_vram_mib": (item.peak_vram_mib for item in telemetry_items),
            "avg_gpu_util_pct": (item.avg_gpu_util_pct for item in telemetry_items),
            "avg_power_w": (item.avg_power_w for item in telemetry_items),
        }
        for metric, values in telemetry_values.items():
            add_statistics(
                row,
                metric,
                values,
                bootstrap_samples=bootstrap_samples,
                seed=stable_seed(bootstrap_seed, engine, batch_size, metric),
            )

        known_counts = [
            item
            for item in items
            if item.requested_requests is not None and item.failed_requests is not None
        ]
        row["error_rate_known_runs"] = len(known_counts)
        if len(known_counts) == len(items) and items:
            requested = sum(item.requested_requests or 0 for item in known_counts)
            failed = sum(item.failed_requests or 0 for item in known_counts)
            completed = sum(item.completed_requests or 0 for item in known_counts)
            row.update(
                {
                    "requested_requests": requested,
                    "completed_requests": completed,
                    "failed_requests": failed,
                    "error_rate": failed / requested if requested else None,
                }
            )
        else:
            row.update(
                {
                    "requested_requests": None,
                    "completed_requests": None,
                    "failed_requests": None,
                    "error_rate": None,
                }
            )
        add_statistics(
            row,
            "run_error_rate",
            (item.error_rate for item in items),
            bootstrap_samples=bootstrap_samples,
            seed=stable_seed(bootstrap_seed, engine, batch_size, "error_rate"),
        )
        rows.append(row)

    baselines = {
        row["engine"]: row.get("output_throughput_median")
        for row in rows
        if row["batch_size"] == 1
        and row.get("output_throughput_median") is not None
        and row.get("error_rate") == 0.0
    }
    for row in rows:
        baseline = baselines.get(row["engine"])
        throughput = row.get("output_throughput_median")
        if baseline is not None and baseline > 0 and throughput is not None:
            speedup = throughput / baseline
            row["speedup_vs_batch_1"] = speedup
            row["scaling_efficiency"] = speedup / row["batch_size"]
        else:
            row["speedup_vs_batch_1"] = None
            row["scaling_efficiency"] = None
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "engine",
        "batch_size",
        "expected_repeats",
        "observed_runs",
        "unique_repeats",
        "telemetry_runs",
    ]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def format_interval(row: dict[str, Any], metric: str, digits: int = 2) -> str:
    median = row.get(f"{metric}_median")
    low = row.get(f"{metric}_ci95_low")
    high = row.get(f"{metric}_ci95_high")
    if median is None:
        return "—"
    if low is None or high is None:
        return format_number(median, digits)
    return (
        f"{format_number(median, digits)} "
        f"[{format_number(low, digits)}; {format_number(high, digits)}]"
    )


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        "| Движок | Batch | Запуски | Output tok/s, median [95% CI] | "
        "Req/s | p95 TTFT, ms | p95 E2E, ms | Ошибки | Peak VRAM, MiB | Speedup | Efficiency |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| {engine} | {batch} | {runs}/{expected} | {throughput} | {rps} | "
            "{ttft} | {e2e} | {errors} | {vram} | {speedup} | {efficiency} |".format(
                engine=row["engine"],
                batch=row["batch_size"],
                runs=row["unique_repeats"],
                expected=row["expected_repeats"],
                throughput=format_interval(row, "output_throughput"),
                rps=format_number(row.get("request_throughput_median")),
                ttft=format_number(row.get("p95_ttft_ms_median")),
                e2e=format_number(row.get("p95_e2e_latency_ms_median")),
                errors=(
                    f"{100.0 * row['error_rate']:.2f}%"
                    if row.get("error_rate") is not None
                    else "—"
                ),
                vram=format_number(row.get("peak_vram_mib_median"), 1),
                speedup=format_number(row.get("speedup_vs_batch_1"), 3),
                efficiency=format_number(row.get("scaling_efficiency"), 3),
            )
        )
    return "\n".join(lines)


def run_key_list(items: Sequence[dict[str, Any]], limit: int = 20) -> str:
    if not items:
        return "нет"
    labels = [
        f"{item['engine']}/b{item['batch_size']}/r{item['repeat']}"
        + (f" (x{item['count']})" if item.get("count") else "")
        for item in items[:limit]
    ]
    suffix = f" и ещё {len(items) - limit}" if len(items) > limit else ""
    return ", ".join(labels) + suffix


def throughput_winners(rows: Sequence[dict[str, Any]]) -> list[str]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["batch_size"])].append(row)
    winners = []
    for batch_size, candidates in sorted(grouped.items()):
        stable = [
            row
            for row in candidates
            if row.get("output_throughput_median") is not None
            and row.get("error_rate") == 0.0
            and row.get("error_rate_known_runs") == row.get("observed_runs")
        ]
        if not stable:
            winners.append(f"batch {batch_size}: нет конфигурации без ошибок")
            continue
        winner = max(stable, key=lambda row: row["output_throughput_median"])
        winners.append(
            f"batch {batch_size}: {winner['engine']} "
            f"({winner['output_throughput_median']:.2f} output tok/s)"
        )
    return winners


def batch_one_latency_winners(rows: Sequence[dict[str, Any]]) -> list[str]:
    candidates = [
        row
        for row in rows
        if row.get("batch_size") == 1
        and row.get("error_rate") == 0.0
        and row.get("error_rate_known_runs") == row.get("observed_runs")
    ]
    winners: list[str] = []
    for metric, label in (
        ("p95_ttft_ms_median", "p95 TTFT"),
        ("p95_e2e_latency_ms_median", "p95 E2E"),
    ):
        measured = [row for row in candidates if row.get(metric) is not None]
        if not measured:
            continue
        winner = min(measured, key=lambda row: row[metric])
        winners.append(f"{label}: {winner['engine']} ({winner[metric]:.2f} ms)")
    return winners


def build_report(
    summary: Sequence[dict[str, Any]],
    validation: dict[str, Any],
    *,
    engines: Sequence[str],
    batch_sizes: Sequence[int],
    repeats: int,
    bootstrap_samples: int,
    metadata: dict[str, str],
    plot_paths: Sequence[Path],
) -> str:
    status = validation["status"]
    model_id = metadata.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    model_revision = metadata.get("MODEL_REVISION", "не указан")
    workload = metadata.get("WORKLOAD", "generated-shared-prefix")
    input_length = metadata.get("INPUT_LENGTH", metadata.get("INPUT_TOKENS", "512"))
    output_length = metadata.get("OUTPUT_LENGTH", metadata.get("OUTPUT_TOKENS", "128"))
    num_prompts = metadata.get("NUM_PROMPTS", "не указано")
    warmup_requests = metadata.get(
        "WARMUP_REQUESTS", metadata.get("WARMUP_PROMPTS", "не указано")
    )
    temperature = metadata.get("TEMPERATURE", "0")
    top_p = metadata.get("TOP_P", "1")
    generation_seed = metadata.get("GENERATION_SEED", "не задан")
    ignore_eos = metadata.get("IGNORE_EOS", "false")
    selected_engine = metadata.get("SELECTED_ENGINE")
    gpu_name = metadata.get("GPU_NAME", "NVIDIA GPU")
    gpu_memory = metadata.get("GPU_MEMORY_MIB")
    compute_capability = metadata.get("GPU_COMPUTE_CAPABILITY")
    gpu_description = gpu_name
    if gpu_memory:
        gpu_description += f", {gpu_memory} MiB"
    if compute_capability:
        gpu_description += f", compute capability {compute_capability}"
    lines = [
        "# Сравнение инференс-движков LLM",
        "",
        f"**Статус эксперимента: `{status}`.**",
        "",
    ]
    if status == "not_run":
        lines.extend(
            [
                "В каталоге `raw` нет ни одного валидного результата benchmark. "
                "Производительность не измерялась, поэтому числовые результаты и вывод о лучшем движке отсутствуют.",
                "",
                "Файлы `summary.csv`, `summary.json` и `experiment_metadata.json` созданы только для "
                "фиксации состояния `not_run`; сводка не содержит подставленных значений.",
                "",
            ]
        )
    elif status == "incomplete":
        lines.extend(
            [
                "Матрица запусков заполнена не полностью. Таблица ниже является промежуточной; "
                "по ней нельзя делать окончательный вывод о победителе.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Матрица запусков заполнена полностью. Все итоговые значения рассчитаны только из raw JSONL и telemetry CSV.",
                "",
            ]
        )

    lines.extend(
        [
            "## Что сравнивается",
            "",
            "- **SGLang** — движок с RadixAttention/RadixCache, ориентированный на эффективное переиспользование общих префиксов и сложные serving-сценарии.",
            "- **vLLM** — зрелый универсальный LLM-serving стек с PagedAttention, continuous batching и широким OpenAI-compatible API.",
            "- **LMDeploy / TurboMind** — движок с persistent batching, собственными CUDA kernels и KV cache manager.",
            "",
            "Внутренние оптимизации движков не отключаются: они являются предметом сравнения. "
            "Квантование, speculative decoding, LoRA и multi-GPU tensor parallelism не используются.",
            "",
            "## Зафиксированные условия",
            "",
            "| Параметр | Значение |",
            "|---|---|",
            f"| Модель | `{model_id}` |",
            f"| Revision | `{model_revision}` |",
            "| Формат весов | FP16, без квантования |",
            f"| GPU / tensor parallelism | {gpu_description}, TP=1 |",
            f"| Workload | `{workload}` |",
            f"| Вход / выход | примерно {input_length} / {output_length} токенов |",
            f"| Запросов на ячейку | {num_prompts} |",
            f"| Warm-up запросов перед каждой ячейкой | {warmup_requests} |",
            f"| Sampling | `temperature={temperature}`, `top_p={top_p}`, "
            f"`generation_seed={generation_seed}`, `ignore_eos={ignore_eos}` |",
            "",
            "Нагрузка имитирует RAG: у запросов есть общий 128-токенный системный префикс и "
            "различающаяся 384-токенная часть с контекстом и вопросом. Seed генератора "
            "тестовых запросов меняется между batch/repeat, но остаётся одинаковым для "
            "трёх движков в одной ячейке.",
            "",
            "`Batch size` здесь означает максимальное число одновременно выполняемых запросов "
            "(`max_concurrency`). Это корректнее статического batch для online serving, где все три "
            "движка используют continuous/persistent batching.",
            "",
        ]
    )

    lines.extend(
        [
            "## Протокол",
            "",
            f"- Движки: {', '.join(engines)}.",
            f"- Effective batch size / concurrency: {', '.join(map(str, batch_sizes))}.",
            f"- Повторений для каждой комбинации: {repeats}.",
            f"- Bootstrap: {bootstrap_samples} выборок; 95% CI построен для медианы по независимым повторам.",
            "- Mean, median и sample standard deviation считаются между повторами, а не между отдельными запросами внутри одного запуска.",
            "- Error rate рассчитывается только при известном полном числе отправленных и завершённых запросов.",
            "",
            "## Полнота данных",
            "",
            f"- Ожидалось запусков: {validation['expected_runs']}.",
            f"- Валидных raw-записей: {validation['raw_records']} ({validation['unique_raw_runs']} уникальных ключей).",
            f"- Telemetry-записей: {validation['telemetry_records']} ({validation['unique_telemetry_runs']} уникальных ключей).",
            f"- Отсутствуют raw: {run_key_list(validation['missing_raw_runs'])}.",
            f"- Дубликаты raw: {run_key_list(validation['duplicate_raw_runs'])}.",
            f"- Отсутствует telemetry: {run_key_list(validation['missing_telemetry_runs'])}.",
            f"- Дубликаты telemetry: {run_key_list(validation['duplicate_telemetry_runs'])}.",
            f"- Некорректные raw-записи: {len(validation['invalid_run_data'])}.",
            "",
        ]
    )

    if validation["invalid_run_data"]:
        lines.extend(["### Некорректные измерения", ""])
        for item in validation["invalid_run_data"]:
            label = f"{item['engine']}/b{item['batch_size']}/r{item['repeat']}"
            lines.append(f"- {label}: {'; '.join(item['issues'])}.")
        lines.append("")

    issues = list(validation["raw_issues"]) + list(validation["telemetry_issues"])
    if issues:
        lines.extend(["### Замечания к входным файлам", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")

    if summary:
        lines.extend(["## Результаты", "", markdown_table(summary), ""])
    if plot_paths:
        lines.extend(["## Графики", ""])
        lines.extend(f"- [{path.name}]({path.as_posix()})" for path in plot_paths)
        lines.append("")

    lines.extend(["## Вывод", ""])
    winners = throughput_winners(summary)
    latency_winners = batch_one_latency_winners(summary)
    if status == "complete" and winners:
        lines.append("Победители по медианной пропускной способности:")
        lines.append("")
        lines.extend(f"- {winner}." for winner in winners)
        if latency_winners:
            lines.extend(["", "Минимальная интерактивная задержка при batch=1:", ""])
            lines.extend(f"- {winner}." for winner in latency_winners)
        lines.extend(
            [
                "",
                "Для выбора движка в интерактивном RAG нужно одновременно учитывать output throughput, "
                "p95 TTFT, p95 E2E, error rate и расход VRAM. Максимальный throughput сам по себе не означает лучший пользовательский режим.",
                "",
            ]
        )
    elif status == "not_run":
        lines.extend(
            [
                "Эксперимент ещё не выполнен. Сравнивать SGLang, vLLM и LMDeploy по производительности пока нельзя.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Окончательный победитель не определяется до заполнения всей матрицы и устранения замечаний к данным.",
                "",
            ]
        )
    lines.append(
        "Результаты относятся только к зафиксированным модели, GPU, версиям ПО и workload из metadata; "
        "они не доказывают универсальное превосходство одного движка."
    )
    integration_text = (
        f"По результатам полной матрицы выбран `{selected_engine}`. Он подключён в "
        "приложение как опциональный OpenAI-compatible provider; Yandex AI Studio "
        "остаётся вариантом по умолчанию для запуска без локальной NVIDIA GPU."
        if selected_engine
        else "Benchmark сам не меняет генератор приложения. Основной RAG продолжает "
        "работать через Yandex AI Studio API, пока выбранный движок не будет подключён."
    )
    lines.extend(
        [
            "",
            "## Интеграция в RAG",
            "",
            integration_text,
            "",
            "## Ограничения",
            "",
            "- Peak VRAM относится к окну измеряемого запуска. Средние GPU utilization и power включают короткий запуск benchmark-клиента и считаются диагностическими, а не основой выбора победителя.",
            "- Параметры памяти движков имеют разную семантику: `mem-fraction-static`, `gpu-memory-utilization` и `cache-max-entry-count` нельзя трактовать как один и тот же memory budget. Их значения отдельно сохранены в metadata.",
            "- Ватты показывают среднюю мощность, а не энергию. Для оценки стоимости энергии потребовались бы J/request или J/output-token.",
            "- Результат RAG-like workload нельзя автоматически переносить на другую модель, GPU, длину контекста или совершенно другой профиль запросов.",
            "- На Tesla T4 (SM75) SGLang использовал совместимые Triton/CUDA kernels: быстрый CuTe DSL путь текущей версии рассчитан на более новые GPU. Результат SGLang нельзя напрямую переносить на A100/H100.",
            "",
            "## Воспроизведение",
            "",
            "Нужен Linux-хост с NVIDIA GPU не менее 12 GB VRAM, Docker Engine, NVIDIA Container Toolkit, Compose v2 и примерно 70 GB свободного места.",
            "",
            "```bash",
            "bash scripts/inference_engines/preflight.sh",
            "bash scripts/inference_engines/run_benchmark.sh all",
            "```",
            "",
            "## Официальные источники",
            "",
            "- [SGLang: benchmark online serving](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md)",
            "- [vLLM: OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)",
            "- [LMDeploy: архитектура TurboMind](https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html)",
        ]
    )
    if metadata:
        lines.extend(["", "## Metadata runner", ""])
        for key, value in sorted(metadata.items()):
            lines.append(f"- `{key}={value}`")
    return "\n".join(lines) + "\n"


def svg_line_chart(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    title: str,
    y_label: str,
) -> str | None:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = safe_float(row.get(metric))
        if value is not None:
            series[str(row["engine"])].append((int(row["batch_size"]), value))
    if not series:
        return None
    batches = sorted({batch for points in series.values() for batch, _ in points})
    values = [value for points in series.values() for _, value in points]
    if not batches or not values:
        return None

    width, height = 960, 560
    left, right, top, bottom = 90, 35, 65, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(values) * 1.10 if max(values) > 0 else 1.0
    x_positions = {
        batch: left + (plot_width * index / max(1, len(batches) - 1))
        for index, batch in enumerate(batches)
    }

    def y_position(value: float) -> float:
        return top + plot_height - (value / y_max) * plot_height

    colors = {"sglang": "#2563eb", "vllm": "#dc2626", "lmdeploy": "#059669"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{value:.1f}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827"/>',
        ]
    )
    for batch in batches:
        x = x_positions[batch]
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="13">{batch}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="14">Effective batch size / concurrency</text>'
    )
    parts.append(
        f'<text transform="translate(22 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>'
    )

    for legend_index, (engine, points) in enumerate(sorted(series.items())):
        points = sorted(points)
        color = colors.get(engine, "#7c3aed")
        coordinates = " ".join(
            f"{x_positions[batch]:.2f},{y_position(value):.2f}"
            for batch, value in points
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for batch, value in points:
            parts.append(
                f'<circle cx="{x_positions[batch]:.2f}" cy="{y_position(value):.2f}" r="4" fill="{color}"/>'
            )
        legend_x = left + legend_index * 150
        parts.append(
            f'<line x1="{legend_x}" y1="{height - 8}" x2="{legend_x + 24}" y2="{height - 8}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 31}" y="{height - 4}" font-family="sans-serif" font-size="13">{html.escape(engine)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def generate_plots(rows: Sequence[dict[str, Any]], plots_dir: Path) -> list[Path]:
    definitions = (
        (
            "output_throughput_median",
            "throughput_vs_batch.svg",
            "Output throughput vs batch",
            "Output tokens/s",
        ),
        (
            "p95_ttft_ms_median",
            "p95_ttft_vs_batch.svg",
            "p95 TTFT vs batch",
            "p95 TTFT, ms",
        ),
        (
            "p95_e2e_latency_ms_median",
            "p95_e2e_vs_batch.svg",
            "p95 E2E latency vs batch",
            "p95 E2E, ms",
        ),
        (
            "peak_vram_mib_median",
            "peak_vram_vs_batch.svg",
            "Peak VRAM vs batch",
            "Peak VRAM, MiB",
        ),
    )
    created: list[Path] = []
    for metric, filename, title, y_label in definitions:
        svg = svg_line_chart(rows, metric=metric, title=title, y_label=y_label)
        if svg is None:
            continue
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / filename
        path.write_text(svg, encoding="utf-8")
        created.append(path)
    return created


def metadata_value(metadata: dict[str, str], *names: str) -> str | None:
    upper = {key.upper(): value for key, value in metadata.items()}
    for name in names:
        value = upper.get(name.upper())
        if value and value != "<redacted>":
            return value
    return None


def resolve_configuration(
    metadata: dict[str, str],
    *,
    engines: Sequence[str] | None,
    batch_sizes: Sequence[int] | None,
    repeats: int | None,
    num_prompts: int | None,
) -> tuple[tuple[str, ...], tuple[int, ...], int, int | None]:
    if engines is None:
        value = metadata_value(metadata, "ENGINES", "ENGINE_ORDER")
        engines = parse_string_sequence(value) if value else DEFAULT_ENGINES
    normalized_engines = tuple(normalize_engine(engine) or engine for engine in engines)
    if batch_sizes is None:
        value = metadata_value(metadata, "BATCH_SIZES", "BATCHES")
        batch_sizes = parse_int_sequence(value) if value else DEFAULT_BATCH_SIZES
    if repeats is None:
        value = metadata_value(metadata, "REPEATS", "NUM_REPEATS")
        repeats = int(value) if value else DEFAULT_REPEATS
    if num_prompts is None:
        value = metadata_value(metadata, "NUM_PROMPTS", "PROMPTS", "REQUESTS_PER_RUN")
        num_prompts = int(value) if value else None
    if repeats <= 0 or (num_prompts is not None and num_prompts <= 0):
        raise ValueError("repeats and num_prompts must be positive")
    return (
        normalized_engines,
        tuple(int(value) for value in batch_sizes),
        repeats,
        num_prompts,
    )


def aggregate_run_dir(
    run_dir: Path,
    *,
    engines: Sequence[str] | None = None,
    batch_sizes: Sequence[int] | None = None,
    repeats: int | None = None,
    num_prompts: int | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_telemetry: bool = True,
    report_path: Path | None = None,
    report_plot_prefix: str = "plots",
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_metadata = parse_env_file(run_dir / "run_metadata.env")
    engines, batch_sizes, repeats, num_prompts = resolve_configuration(
        runner_metadata,
        engines=engines,
        batch_sizes=batch_sizes,
        repeats=repeats,
        num_prompts=num_prompts,
    )

    runs, raw_issues = load_benchmark_runs(
        run_dir / "raw",
        expected_requests=num_prompts,
    )
    telemetry, telemetry_issues = load_telemetry_runs(run_dir / "telemetry")
    validation = validate_completeness(
        runs,
        telemetry,
        engines=engines,
        batch_sizes=batch_sizes,
        repeats=repeats,
        raw_issues=raw_issues,
        telemetry_issues=telemetry_issues,
        require_telemetry=require_telemetry,
        expected_requests=num_prompts,
    )
    summary = summarize_runs(
        runs,
        telemetry,
        expected_repeats=repeats,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    plot_paths = generate_plots(summary, run_dir / "plots") if summary else []

    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    metadata_json = run_dir / "experiment_metadata.json"
    report_path = report_path.resolve() if report_path else run_dir / "report.md"
    write_summary_csv(summary_csv, summary)
    write_json(
        summary_json,
        {
            "status": validation["status"],
            "generated_at": utc_now(),
            "rows": summary,
        },
    )
    experiment_metadata = {
        "generated_at": utc_now(),
        "status": validation["status"],
        "run_dir": str(run_dir),
        "configuration": {
            "engines": list(engines),
            "batch_sizes": list(batch_sizes),
            "repeats": repeats,
            "num_prompts": num_prompts,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_statistic": "median",
            "telemetry_required": require_telemetry,
        },
        "validation": validation,
        "runner_metadata": runner_metadata,
        "artifacts": {
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "report": str(report_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    write_json(metadata_json, experiment_metadata)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            summary,
            validation,
            engines=engines,
            batch_sizes=batch_sizes,
            repeats=repeats,
            bootstrap_samples=bootstrap_samples,
            metadata=runner_metadata,
            plot_paths=[Path(report_plot_prefix) / path.name for path in plot_paths],
        ),
        encoding="utf-8",
    )
    return experiment_metadata


def command_result(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error), "stdout": "", "stderr": ""}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_preflight() -> dict[str, Any]:
    nvidia_path = shutil.which("nvidia-smi")
    docker_path = shutil.which("docker")
    nvidia = (
        command_result(
            [
                nvidia_path,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if nvidia_path
        else {"ok": False, "error": "nvidia-smi is not available"}
    )
    docker = (
        command_result([docker_path, "info", "--format", "{{json .ServerVersion}}"])
        if docker_path
        else {"ok": False, "error": "docker is not available"}
    )
    reasons = []
    if not nvidia.get("ok"):
        reasons.append("NVIDIA GPU/CUDA is unavailable")
    if not docker.get("ok"):
        reasons.append("Docker daemon is unavailable")
    return {
        "generated_at": utc_now(),
        "status": "ready" if not reasons else "blocked",
        "reasons": reasons,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "nvidia_smi": nvidia,
        "docker": docker,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate raw benchmark results"
    )
    aggregate.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    aggregate.add_argument("--engines", type=parse_string_sequence)
    aggregate.add_argument("--batch-sizes", type=parse_int_sequence)
    aggregate.add_argument("--repeats", type=int)
    aggregate.add_argument("--num-prompts", type=int)
    aggregate.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    aggregate.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    aggregate.add_argument("--allow-missing-telemetry", action="store_true")
    aggregate.add_argument("--report", type=Path)
    aggregate.add_argument("--report-plot-prefix", default="plots")

    preflight = subparsers.add_parser(
        "preflight", help="Check GPU and Docker availability"
    )
    preflight.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = collect_preflight()
        output = args.run_dir.resolve() / "preflight.json"
        write_json(output, result)
        print(f"Preflight status: {result['status']}")
        print(f"Result: {output}")
        return 0 if result["status"] == "ready" else 2

    result = aggregate_run_dir(
        args.run_dir,
        engines=args.engines,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        num_prompts=args.num_prompts,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        require_telemetry=not args.allow_missing_telemetry,
        report_path=args.report,
        report_plot_prefix=args.report_plot_prefix,
    )
    print(f"Experiment status: {result['status']}")
    print(f"Summary: {result['artifacts']['summary_csv']}")
    print(f"Report: {result['artifacts']['report']}")
    return 0 if result["status"] in {"complete", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
