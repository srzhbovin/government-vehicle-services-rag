import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from priority_gateway.token_routing_benchmark import (  # noqa: E402
    ReplicaState,
    RequestProfile,
    actual_service_seconds,
    calibrate_ewma,
    measure_assignments,
    route_workload,
    run_experiment,
    summarize,
    write_results,
)


def request(request_id: str, input_tokens: int = 100, output_tokens: int = 100):
    return RequestProfile(
        request_id=request_id,
        request_class="faq",
        input_tokens=input_tokens,
        actual_output_tokens=output_tokens,
        max_output_tokens=512,
    )


class TokenAwareRoutingTests(unittest.TestCase):
    def test_decode_token_is_more_expensive_than_prefill_token(self):
        replica = ReplicaState("r", 6000, 300, 32768)
        base = actual_service_seconds(request("base"), replica)
        extra_input = actual_service_seconds(
            request("input", input_tokens=200), replica
        )
        extra_output = actual_service_seconds(
            request("output", output_tokens=200), replica
        )

        self.assertGreater(extra_output - base, extra_input - base)

    def test_token_router_selects_replica_with_smallest_residual_work(self):
        replicas = route_workload(
            [request("one")],
            "token_ewma",
            replica_count=3,
            ewma_estimates=calibrate_ewma(),
            initial_token_loads=[15000, 4000, 9000],
        )

        selected = next(replica.name for replica in replicas if replica.assignments)

        self.assertEqual(selected, "replica-2")

    def test_token_ewma_improves_makespan_under_residual_load(self):
        workload = [request(f"short-{index}", 150, 60) for index in range(18)] + [
            request(f"long-{index}", 3000, 900) for index in range(6)
        ]
        ewma = {"faq": 80, "rag": 250, "analysis": 900}
        rows = []
        for strategy in ("round_robin", "token_ewma"):
            replicas = route_workload(
                workload,
                strategy,
                replica_count=3,
                ewma_estimates=ewma,
                initial_token_loads=[15000, 4000, 9000],
            )
            rows.extend(
                measure_assignments(
                    replicas,
                    strategy,
                    len(workload),
                    repeat=1,
                    scenario="residual_load",
                )
            )
        summary = summarize(rows)
        by_strategy = {row["strategy"]: row for row in summary}

        self.assertLess(
            by_strategy["token_ewma"]["makespan_ms"],
            by_strategy["round_robin"]["makespan_ms"],
        )

    def test_full_experiment_is_deterministic_and_writes_six_plots(self):
        args = Namespace(
            batch_sizes=[12],
            repeats=1,
            replicas=3,
            seed=7,
        )
        first = run_experiment(args)
        second = run_experiment(args)
        self.assertEqual(first[1], second[1])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_results(output, *first)

            self.assertEqual(len(list((output / "plots").glob("*.svg"))), 6)
            self.assertTrue((output / "assignments.csv").exists())
            self.assertTrue((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
