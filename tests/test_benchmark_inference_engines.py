import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.benchmark_inference_engines import (  # noqa: E402
    aggregate_run_dir,
    batch_one_latency_winners,
    bootstrap_confidence_interval,
    describe_values,
    load_benchmark_runs,
    load_telemetry_runs,
    parse_env_file,
    throughput_winners,
)


class InferenceEngineBenchmarkTests(unittest.TestCase):
    def test_metadata_redacts_credentials_but_not_token_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_metadata.env"
            path.write_text(
                "HF_TOKEN=secret\nINPUT_TOKENS=512\nOUTPUT_TOKENS=128\n",
                encoding="utf-8",
            )

            values = parse_env_file(path)

            self.assertEqual(values["HF_TOKEN"], "<redacted>")
            self.assertEqual(values["INPUT_TOKENS"], "512")
            self.assertEqual(values["OUTPUT_TOKENS"], "128")

    def test_describe_values_is_deterministic(self):
        first = describe_values([1.0, 2.0, 3.0], bootstrap_samples=500, seed=7)
        second = describe_values([1.0, 2.0, 3.0], bootstrap_samples=500, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first["n"], 3)
        self.assertEqual(first["mean"], 2.0)
        self.assertEqual(first["median"], 2.0)
        self.assertAlmostEqual(first["std"], 1.0)
        self.assertLessEqual(first["ci95_low"], first["median"])
        self.assertGreaterEqual(first["ci95_high"], first["median"])

    def test_bootstrap_empty_and_single_value(self):
        self.assertEqual(bootstrap_confidence_interval([]), (None, None))
        self.assertEqual(bootstrap_confidence_interval([4.25]), (4.25, 4.25))

    def test_loads_official_raw_and_derives_older_p95_ttft(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            record = {
                "backend": "sglang-oai",
                "max_concurrency": 2,
                "tag": "repeat=1",
                "duration": 4.0,
                "completed": 3,
                "request_throughput": 0.75,
                "input_throughput": 120.0,
                "output_throughput": 24.0,
                "total_throughput": 144.0,
                "mean_ttft_ms": 100.0,
                "median_ttft_ms": 95.0,
                "p99_ttft_ms": 150.0,
                "ttfts": [0.08, 0.10, 0.14],
                "input_lens": [100, 100, 100, 100],
                "errors": ["", "", "timeout", ""],
            }
            (raw_dir / "results.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            runs, issues = load_benchmark_runs(raw_dir)

            self.assertEqual(issues, [])
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.key.engine, "sglang")
            self.assertEqual(run.key.batch_size, 2)
            self.assertEqual(run.key.repeat, 1)
            self.assertEqual(run.requested_requests, 4)
            self.assertEqual(run.failed_requests, 1)
            self.assertEqual(run.error_rate, 0.25)
            self.assertAlmostEqual(run.metrics["p95_ttft_ms"], 136.0)

    def test_loads_nvidia_smi_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_dir = Path(directory)
            path = telemetry_dir / "vllm_batch_4_repeat_2.csv"
            path.write_text(
                "timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"
                "2026/07/11 12:00:00, 50 %, 4000 MiB, 100 W\n"
                "2026/07/11 12:00:01, 70 %, 4500 MiB, 120 W\n",
                encoding="utf-8",
            )

            rows, issues = load_telemetry_runs(telemetry_dir)

            self.assertEqual(issues, [])
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.key.engine, "vllm")
            self.assertEqual(row.key.batch_size, 4)
            self.assertEqual(row.key.repeat, 2)
            self.assertEqual(row.samples, 2)
            self.assertEqual(row.peak_vram_mib, 4500.0)
            self.assertEqual(row.avg_gpu_util_pct, 60.0)
            self.assertEqual(row.avg_power_w, 110.0)

    def test_complete_aggregation_writes_statistics_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "raw").mkdir()
            (run_dir / "telemetry").mkdir()
            (run_dir / "run_metadata.env").write_text(
                "ENGINE_ORDER=sglang vllm\n"
                "BATCH_SIZES=1,2\n"
                "REPEATS=2\n"
                "NUM_PROMPTS=4\n"
                "MODEL=Qwen/Qwen2.5-3B-Instruct\n",
                encoding="utf-8",
            )
            base_throughput = {"sglang": {1: 10.0, 2: 18.0}, "vllm": {1: 9.0, 2: 15.0}}
            for engine in ("sglang", "vllm"):
                for batch in (1, 2):
                    for repeat in (1, 2):
                        throughput = base_throughput[engine][batch] + (repeat - 1) * 2.0
                        raw = {
                            "backend": engine,
                            "max_concurrency": batch,
                            "tag": f"repeat={repeat}",
                            "duration": 2.0,
                            "completed": 4,
                            "request_throughput": throughput / 10,
                            "input_throughput": throughput * 4,
                            "output_throughput": throughput,
                            "total_throughput": throughput * 5,
                            "p95_ttft_ms": 100.0 * batch + repeat,
                            "p95_e2e_latency_ms": 1000.0 * batch + repeat,
                            "p95_tpot_ms": 20.0 + repeat,
                            "errors": ["", "", "", ""],
                        }
                        raw_path = (
                            run_dir
                            / "raw"
                            / f"{engine}_batch_{batch}_repeat_{repeat}.jsonl"
                        )
                        raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
                        telemetry_path = (
                            run_dir
                            / "telemetry"
                            / f"{engine}_batch_{batch}_repeat_{repeat}.csv"
                        )
                        telemetry_path.write_text(
                            "timestamp, utilization.gpu [%], memory.used [MiB], power.draw [W]\n"
                            f"t1, 50 %, {3000 + batch * 100} MiB, 100 W\n"
                            f"t2, 70 %, {3200 + batch * 100} MiB, 120 W\n",
                            encoding="utf-8",
                        )

            metadata = aggregate_run_dir(
                run_dir,
                bootstrap_samples=300,
                bootstrap_seed=5,
                report_plot_prefix="inference_engine_plots",
            )

            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["validation"]["missing_raw_runs"], [])
            summary_document = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary_document["status"], "complete")
            self.assertEqual(len(summary_document["rows"]), 4)
            sglang_batch_2 = next(
                row
                for row in summary_document["rows"]
                if row["engine"] == "sglang" and row["batch_size"] == 2
            )
            self.assertEqual(sglang_batch_2["output_throughput_mean"], 19.0)
            self.assertEqual(sglang_batch_2["output_throughput_median"], 19.0)
            self.assertEqual(sglang_batch_2["error_rate"], 0.0)
            self.assertAlmostEqual(sglang_batch_2["speedup_vs_batch_1"], 19.0 / 11.0)
            self.assertAlmostEqual(
                sglang_batch_2["scaling_efficiency"],
                (19.0 / 11.0) / 2.0,
            )
            self.assertEqual(sglang_batch_2["peak_vram_mib_median"], 3400.0)
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "experiment_metadata.json").exists())
            self.assertTrue((run_dir / "report.md").exists())
            self.assertTrue((run_dir / "plots" / "throughput_vs_batch.svg").exists())
            self.assertTrue((run_dir / "plots" / "p95_ttft_vs_batch.svg").exists())
            self.assertTrue((run_dir / "plots" / "p95_e2e_vs_batch.svg").exists())
            self.assertTrue((run_dir / "plots" / "peak_vram_vs_batch.svg").exists())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("inference_engine_plots/throughput_vs_batch.svg", report)

    def test_empty_raw_produces_not_run_without_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metadata = aggregate_run_dir(
                run_dir,
                engines=("sglang",),
                batch_sizes=(1,),
                repeats=1,
                bootstrap_samples=50,
            )

            self.assertEqual(metadata["status"], "not_run")
            self.assertEqual(
                json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
                    "rows"
                ],
                [],
            )
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertFalse((run_dir / "plots").exists())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("`not_run`", report)
            self.assertIn("не содержит подставленных значений", report)

    def test_missing_cell_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "raw").mkdir()
            (run_dir / "telemetry").mkdir()
            raw = {
                "backend": "sglang",
                "max_concurrency": 1,
                "tag": "repeat=1",
                "completed": 1,
                "output_throughput": 10.0,
                "errors": [""],
            }
            (run_dir / "raw" / "sglang_batch_1_repeat_1.jsonl").write_text(
                json.dumps(raw) + "\n",
                encoding="utf-8",
            )
            (run_dir / "telemetry" / "sglang_batch_1_repeat_1.csv").write_text(
                "memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
                "3000 MiB, 50 %, 90 W\n",
                encoding="utf-8",
            )

            metadata = aggregate_run_dir(
                run_dir,
                engines=("sglang",),
                batch_sizes=(1, 2),
                repeats=1,
                bootstrap_samples=50,
            )

            self.assertEqual(metadata["status"], "incomplete")
            self.assertEqual(
                metadata["validation"]["missing_raw_runs"],
                [{"engine": "sglang", "batch_size": 2, "repeat": 1}],
            )

    def test_short_b_and_r_filename_identifiers_are_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            for repeat in (1, 2, 3):
                record = {
                    "backend": "sglang-oai",
                    "max_concurrency": 1,
                    "completed": 1,
                    "output_throughput": 10.0 + repeat,
                    "errors": [""],
                }
                (raw_dir / f"sglang_b1_r{repeat}.jsonl").write_text(
                    json.dumps(record) + "\n",
                    encoding="utf-8",
                )

            runs, issues = load_benchmark_runs(raw_dir)

            self.assertEqual(issues, [])
            self.assertEqual([run.key.repeat for run in runs], [1, 2, 3])

    def test_complete_matrix_with_wrong_request_count_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "raw").mkdir()
            (run_dir / "telemetry").mkdir()
            (run_dir / "run_metadata.env").write_text(
                "ENGINE_ORDER=sglang\nBATCH_SIZES=1\nREPEATS=1\nNUM_PROMPTS=4\n",
                encoding="utf-8",
            )
            raw = {
                "backend": "sglang-oai",
                "max_concurrency": 1,
                "tag": "repeat=1",
                "duration": 2.0,
                "completed": 3,
                "request_throughput": 1.5,
                "output_throughput": 120.0,
                "p95_ttft_ms": 50.0,
                "p95_tpot_ms": 10.0,
                "p95_e2e_latency_ms": 500.0,
                "errors": ["", "", ""],
            }
            (run_dir / "raw" / "sglang_b1_r1.jsonl").write_text(
                json.dumps(raw) + "\n",
                encoding="utf-8",
            )
            (run_dir / "telemetry" / "sglang_b1_r1.csv").write_text(
                "memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
                "3000 MiB, 50 %, 90 W\n",
                encoding="utf-8",
            )

            metadata = aggregate_run_dir(run_dir, bootstrap_samples=50)

            self.assertEqual(metadata["status"], "incomplete")
            invalid = metadata["validation"]["invalid_run_data"]
            self.assertEqual(len(invalid), 1)
            self.assertIn("expected 4 requests, got 3", invalid[0]["issues"])

    def test_throughput_winner_excludes_rows_with_errors(self):
        rows = [
            {
                "engine": "sglang",
                "batch_size": 4,
                "output_throughput_median": 200.0,
                "error_rate": 0.1,
                "error_rate_known_runs": 5,
                "observed_runs": 5,
            },
            {
                "engine": "vllm",
                "batch_size": 4,
                "output_throughput_median": 180.0,
                "error_rate": 0.0,
                "error_rate_known_runs": 5,
                "observed_runs": 5,
            },
        ]

        winners = throughput_winners(rows)

        self.assertEqual(winners, ["batch 4: vllm (180.00 output tok/s)"])

    def test_batch_one_latency_winners_exclude_failed_engine(self):
        rows = [
            {
                "engine": "sglang",
                "batch_size": 1,
                "p95_ttft_ms_median": 70.0,
                "p95_e2e_latency_ms_median": 900.0,
                "error_rate": 0.2,
                "error_rate_known_runs": 5,
                "observed_runs": 5,
            },
            {
                "engine": "vllm",
                "batch_size": 1,
                "p95_ttft_ms_median": 80.0,
                "p95_e2e_latency_ms_median": 850.0,
                "error_rate": 0.0,
                "error_rate_known_runs": 5,
                "observed_runs": 5,
            },
            {
                "engine": "lmdeploy",
                "batch_size": 1,
                "p95_ttft_ms_median": 75.0,
                "p95_e2e_latency_ms_median": 920.0,
                "error_rate": 0.0,
                "error_rate_known_runs": 5,
                "observed_runs": 5,
            },
        ]

        winners = batch_one_latency_winners(rows)

        self.assertEqual(
            winners,
            ["p95 TTFT: lmdeploy (75.00 ms)", "p95 E2E: vllm (850.00 ms)"],
        )


if __name__ == "__main__":
    unittest.main()
