import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from priority_gateway.app import create_app  # noqa: E402
from priority_gateway.config import (  # noqa: E402
    AuthenticationError,
    ConfigManager,
    ConfigurationError,
    load_config,
)
from priority_gateway.scheduler import PriorityScheduler, QueueFullError  # noqa: E402


POLICY = """
upstream:
  url: http://litellm.test:4000
  api_key: upstream-secret
  timeout_seconds: 10
default_group: standard
authentication:
  required: false
scheduler:
  strategy: priority_aging
  max_concurrency: 1
  max_queue_size: 10
  queue_timeout_seconds: 5
  aging_interval_seconds: 0.01
groups:
  critical:
    priority: 0
  standard:
    priority: 10
  batch:
    priority: 20
rules:
  - group: critical
    match:
      bearer_tokens: ["${CRITICAL_TEST_KEY}"]
  - group: batch
    match:
      headers:
        x-workload-class: offline
"""


class PriorityConfigTests(unittest.TestCase):
    def test_classifies_by_existing_token_or_header_without_exposing_token(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CRITICAL_TEST_KEY": "critical-secret"}),
        ):
            path = Path(directory) / "priorities.yaml"
            path.write_text(POLICY, encoding="utf-8")
            config = load_config(path)

        self.assertEqual(
            config.classify({"authorization": "Bearer critical-secret"}).name,
            "critical",
        )
        self.assertEqual(
            config.classify({"x-workload-class": "offline"}).name,
            "batch",
        )
        self.assertEqual(config.classify({}).name, "standard")
        self.assertNotIn("critical-secret", repr(config.rules[0]))

    def test_rejects_unknown_rule_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priorities.yaml"
            path.write_text(
                POLICY.replace(
                    "group: critical\n    match:", "group: missing\n    match:"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_required_authentication_rejects_unknown_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priorities.yaml"
            path.write_text(
                POLICY.replace("required: false", "required: true"), encoding="utf-8"
            )
            config = load_config(path)

        with self.assertRaises(AuthenticationError):
            config.classify({"authorization": "Bearer unknown"})

    def test_invalid_hot_reload_keeps_last_valid_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priorities.yaml"
            path.write_text(POLICY, encoding="utf-8")
            manager = ConfigManager(path)
            valid = manager.get()
            path.write_text("groups: [invalid", encoding="utf-8")

            fallback = manager.get()

        self.assertIs(fallback, valid)
        self.assertIsNotNone(manager.last_reload_error)


class PrioritySchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scheduler = PriorityScheduler(
            strategy="priority_aging",
            max_concurrency=1,
            max_queue_size=10,
            queue_timeout_seconds=2,
            aging_interval_seconds=1,
        )

    async def test_high_priority_overtakes_earlier_low_priority_request(self):
        blocker = await self.scheduler.acquire("blocker", 0)
        low_task = asyncio.create_task(self.scheduler.acquire("batch", 20))
        await asyncio.sleep(0)
        high_task = asyncio.create_task(self.scheduler.acquire("critical", 0))
        await asyncio.sleep(0)

        await blocker.release()
        high = await high_task
        self.assertFalse(low_task.done())
        await high.release()
        low = await low_task
        await low.release()

    async def test_aging_lets_old_low_priority_overtake_new_high_priority(self):
        scheduler = PriorityScheduler(
            strategy="priority_aging",
            max_concurrency=1,
            max_queue_size=10,
            queue_timeout_seconds=2,
            aging_interval_seconds=0.01,
        )
        blocker = await scheduler.acquire("blocker", 0)
        old_low_task = asyncio.create_task(scheduler.acquire("batch", 2))
        await asyncio.sleep(0.031)
        new_high_task = asyncio.create_task(scheduler.acquire("critical", 0))
        await asyncio.sleep(0)

        await blocker.release()
        old_low = await old_low_task
        self.assertFalse(new_high_task.done())
        await old_low.release()
        new_high = await new_high_task
        await new_high.release()

    async def test_starvation_timeout_serves_overdue_batch_before_new_critical(self):
        scheduler = PriorityScheduler(
            strategy="priority_aging",
            max_concurrency=1,
            max_queue_size=10,
            queue_timeout_seconds=2,
            aging_interval_seconds=10,
            starvation_timeout_seconds=0.02,
        )
        blocker = await scheduler.acquire("blocker", 0)
        old_batch_task = asyncio.create_task(scheduler.acquire("batch", 20))
        await asyncio.sleep(0.025)
        new_critical_task = asyncio.create_task(scheduler.acquire("critical", 0))
        await asyncio.sleep(0)

        await blocker.release()
        old_batch = await old_batch_task
        self.assertFalse(new_critical_task.done())
        await old_batch.release()
        new_critical = await new_critical_task
        await new_critical.release()

    async def test_bounded_queue_rejects_overload(self):
        scheduler = PriorityScheduler(
            strategy="fifo",
            max_concurrency=1,
            max_queue_size=1,
            queue_timeout_seconds=2,
            aging_interval_seconds=1,
        )
        blocker = await scheduler.acquire("active", 0)
        waiting = asyncio.create_task(scheduler.acquire("standard", 10))
        await asyncio.sleep(0)
        with self.assertRaises(QueueFullError):
            await scheduler.acquire("standard", 10)
        await blocker.release()
        lease = await waiting
        await lease.release()


class PriorityGatewayProxyTests(unittest.TestCase):
    def test_proxies_openai_request_and_preserves_status(self):
        captured = {}

        def upstream(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("authorization")
            return httpx.Response(
                201,
                json={"id": "chatcmpl-test", "choices": []},
                headers={"x-upstream": "litellm"},
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CRITICAL_TEST_KEY": "critical-secret"}),
        ):
            path = Path(directory) / "priorities.yaml"
            path.write_text(POLICY, encoding="utf-8")
            app = create_app(path, transport=httpx.MockTransport(upstream))
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer critical-secret"},
                    json={"model": "dmv-rag", "messages": [], "stream": False},
                )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers["x-priority-group"], "critical")
        self.assertIn("x-queue-time-ms", response.headers)
        self.assertEqual(response.headers["x-upstream"], "litellm")
        self.assertEqual(captured["authorization"], "Bearer upstream-secret")
        self.assertEqual(
            captured["url"], "http://litellm.test:4000/v1/chat/completions"
        )

    def test_streaming_response_releases_scheduler_slot(self):
        def upstream(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b'data: {"delta":"ok"}\n\ndata: [DONE]\n\n',
                headers={"content-type": "text/event-stream"},
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"CRITICAL_TEST_KEY": "critical-secret"}),
        ):
            path = Path(directory) / "priorities.yaml"
            path.write_text(POLICY, encoding="utf-8")
            app = create_app(path, transport=httpx.MockTransport(upstream))
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer critical-secret"},
                    json={"model": "dmv-rag", "messages": [], "stream": True},
                )
                metrics = client.get("/metrics").json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(metrics["active"], 0)
        self.assertEqual(metrics["completed"], 1)


if __name__ == "__main__":
    unittest.main()
