import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import httpx


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.generator import (  # noqa: E402
    GenerationError,
    LocalOpenAICompatibleGenerator,
    YandexGenerator,
    build_context_prompt,
)
from rag_pipeline.prompting import PromptStrategy  # noqa: E402
from rag_pipeline.retriever import RetrievedChunk  # noqa: E402
from rag_pipeline.settings import Settings  # noqa: E402


def settings_with_credentials(**changes):
    settings = Settings.from_env()
    values = {
        "yandex_api_key": "test-secret",
        "yandex_folder_id": "folder-id",
        "yandex_model": "yandexgpt-5-lite",
        "yandex_fallback_models": (),
        "prompt_strategy": "plain",
    }
    values.update(changes)
    return replace(settings, **values)


def source(rank=1):
    return RetrievedChunk(
        rank=rank,
        chunk_index=0,
        chunk_id="doc-1::chunk-0",
        document_id="doc-1",
        title="Renew registration",
        text="You can renew your registration online.",
        score=0.9,
    )


class GeneratorTests(unittest.TestCase):
    def test_context_prompt_contains_question_source_and_marker(self):
        prompt = build_context_prompt("How do I renew?", [source()])

        self.assertIn("How do I renew?", prompt)
        self.assertIn("[1] Title: Renew registration", prompt)
        self.assertIn("doc-1::chunk-0", prompt)

    def test_parses_responses_api_output_and_usage(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "gpt://folder-id/yandexgpt-5-lite",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Renew online [1]."}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 12,
                        "total_tokens": 112,
                    },
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = YandexGenerator(
            settings_with_credentials(),
            http_client=client,
        ).generate("How do I renew?", [source()])

        self.assertEqual(result.answer, "Renew online [1].")
        self.assertEqual(result.usage.total_tokens, 112)
        self.assertEqual(captured["authorization"], "Api-Key test-secret")
        self.assertIn("Answer only", captured["payload"]["instructions"])

    def test_falls_back_when_model_is_not_available(self):
        models = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            models.append(payload["model"])
            if payload["model"].endswith("yandexgpt-5-lite"):
                return httpx.Response(404, json={"error": "model not found"})
            return httpx.Response(200, json={"output_text": "Fallback answer"})

        settings = settings_with_credentials(
            yandex_fallback_models=("yandexgpt-5.1",),
        )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = YandexGenerator(settings, http_client=client).generate(
            "Question",
            [source()],
        )

        self.assertEqual(result.answer, "Fallback answer")
        self.assertEqual(len(models), 2)
        self.assertTrue(models[1].endswith("yandexgpt-5.1"))

    def test_native_structured_output_is_sent_and_parsed(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "gpt://folder-id/yandexgpt-5-lite",
                    "output_text": json.dumps(
                        {
                            "answer": "Renew online [1].",
                            "confidence": 0.95,
                            "source": "[1]",
                        }
                    ),
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = YandexGenerator(
            settings_with_credentials(prompt_strategy="structured_output"),
            http_client=client,
        ).generate("How do I renew?", [source()])

        response_format = captured["payload"]["text"]["format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["strict"])
        self.assertEqual(result.answer, "Renew online [1].")
        self.assertTrue(result.schema_valid)
        self.assertEqual(result.confidence, 0.95)

    def test_does_not_expose_rejected_api_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": "Unknown api key 'test-secret'"},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with self.assertRaises(GenerationError) as context:
            YandexGenerator(
                settings_with_credentials(),
                http_client=client,
            ).generate("Question", [source()])

        self.assertNotIn("test-secret", str(context.exception))
        self.assertIn("API key was rejected", str(context.exception))

    def test_local_generator_uses_available_model_and_language(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "google/gemma-3-4b"}]},
                )
            payload = json.loads(request.content)
            self.assertIn("Answer in Russian", payload["messages"][1]["content"])
            return httpx.Response(
                200,
                json={
                    "model": "google/gemma-3-4b",
                    "choices": [
                        {"message": {"content": "Ответ на русском [1]."}}
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 8,
                        "total_tokens": 58,
                    },
                },
            )

        settings = replace(
            Settings.from_env(),
            llm_provider="lmstudio",
            local_llm_model="missing-model",
            prompt_strategy="plain",
        )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = LocalOpenAICompatibleGenerator(settings, client).generate(
            "Как продлить регистрацию?",
            [source()],
            language="ru",
        )

        self.assertEqual(result.answer, "Ответ на русском [1].")
        self.assertEqual(result.model, "google/gemma-3-4b")
        self.assertEqual(result.usage.total_tokens, 58)
        self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
