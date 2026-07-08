import sys
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from backend.main import create_app  # noqa: E402
from rag_pipeline.generator import GenerationResult, GuardrailResult  # noqa: E402
from rag_pipeline.retriever import (  # noqa: E402
    RetrievalResult,
    RetrievedChunk,
)
from rag_pipeline.schemas import GenerationUsage  # noqa: E402
from rag_pipeline.service import RAGService  # noqa: E402
from rag_pipeline.settings import Settings  # noqa: E402


class FakeRetriever:
    ready = True
    method_name = "hybrid_reranked"

    def __init__(self):
        self.calls = []

    def initialize(self):
        self.ready = True

    def retrieve(
        self,
        query,
        top_k=None,
        context_window=None,
        intro_chunks=None,
        max_context_chunks=None,
    ):
        self.calls.append(
            {
                "top_k": top_k,
                "context_window": context_window,
                "intro_chunks": intro_chunks,
                "max_context_chunks": max_context_chunks,
            }
        )
        chunk = RetrievedChunk(
            rank=1,
            chunk_index=0,
            chunk_id="doc-1::chunk-0",
            document_id="doc-1",
            title="DMV document",
            text="Relevant context",
            score=1.25,
        )
        return RetrievalResult(
            chunks=[chunk],
            method=self.method_name,
            elapsed_ms=12.5,
        )


class FakeGenerator:
    configured = True

    def __init__(self):
        self.generate_calls = 0
        self.judge_calls = 0

    def generate(
        self,
        question,
        chunks,
        language="auto",
        strategy=None,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
    ):
        self.generate_calls += 1
        return GenerationResult(
            answer="Grounded answer [1].",
            model="fake-model",
            usage=GenerationUsage(input_tokens=10, output_tokens=4, total_tokens=14),
            elapsed_ms=20.0,
        )

    def judge_answer(self, question, chunks, answer, language="auto"):
        self.judge_calls += 1
        return GuardrailResult(
            checked=True,
            accepted=True,
            verdict="grounded",
            score=0.95,
            reason="The answer is supported.",
            elapsed_ms=8.0,
            parse_success=True,
            schema_valid=True,
        )


class RejectThenAcceptGenerator(FakeGenerator):
    def generate(
        self,
        question,
        chunks,
        language="auto",
        strategy=None,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
    ):
        self.generate_calls += 1
        answer = (
            "Unsupported compact answer [1]."
            if self.generate_calls == 1
            else "Grounded expanded answer [1]."
        )
        return GenerationResult(
            answer=answer,
            model="fake-model",
            usage=GenerationUsage(input_tokens=10, output_tokens=4, total_tokens=14),
            elapsed_ms=20.0,
        )

    def judge_answer(self, question, chunks, answer, language="auto"):
        self.judge_calls += 1
        if self.judge_calls == 1:
            return GuardrailResult(
                checked=True,
                accepted=False,
                verdict="not_grounded",
                score=0.2,
                reason="The compact context is not enough.",
                elapsed_ms=8.0,
                parse_success=True,
                schema_valid=True,
            )
        return GuardrailResult(
            checked=True,
            accepted=True,
            verdict="grounded",
            score=0.95,
            reason="The expanded answer is supported.",
            elapsed_ms=8.0,
            parse_success=True,
            schema_valid=True,
        )


class RAGApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = replace(
            Settings.from_env(),
            eager_load=False,
            yandex_api_key="test",
            yandex_folder_id="folder",
            top_k=3,
            reranker_candidate_k=20,
            max_context_chunks=6,
            enable_adaptive_context=True,
        )
        cls.fake_retriever = FakeRetriever()
        cls.fake_generator = FakeGenerator()
        service = RAGService(
            settings,
            retriever=cls.fake_retriever,
            generator=cls.fake_generator,
        )
        cls.client_context = TestClient(create_app(service=service, settings=settings))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["retrieval_method"], "hybrid_reranked")
        self.assertEqual(response.json()["prompt_strategy"], "structured_output")
        self.assertTrue(response.json()["judge_enabled"])
        self.assertTrue(response.json()["adaptive_context_enabled"])
        self.assertEqual(response.json()["adaptive_top_k"], 5)

    def test_retrieve_endpoint(self):
        response = self.client.post(
            "/api/v1/retrieve",
            json={"question": "How do I renew?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["document_id"], "doc-1")
        self.assertEqual(response.json()["top_k"], 3)
        self.assertEqual(response.json()["context_window"], 0)
        self.assertEqual(response.json()["intro_chunks"], 0)
        self.assertEqual(response.json()["max_context_chunks"], 6)

    def test_ask_endpoint(self):
        response = self.client.post(
            "/api/v1/ask",
            json={"question": "How do I renew?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Grounded answer [1].")
        self.assertEqual(response.json()["usage"]["total_tokens"], 14)
        self.assertEqual(response.json()["prompt_strategy"], "plain")
        self.assertEqual(response.json()["top_k"], 3)
        self.assertEqual(response.json()["context_window"], 0)
        self.assertEqual(response.json()["intro_chunks"], 0)
        self.assertEqual(response.json()["max_context_chunks"], 6)
        self.assertTrue(response.json()["guardrail"]["accepted"])
        self.assertFalse(response.json()["guardrail"]["corrected"])
        self.assertFalse(response.json()["adaptive_context"]["triggered"])

    def test_ask_endpoint_accepts_context_parameters(self):
        response = self.client.post(
            "/api/v1/ask",
            json={
                "question": "How do I renew?",
                "top_k": 3,
                "context_window": 0,
                "intro_chunks": 1,
                "max_context_chunks": 6,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["top_k"], 3)
        self.assertEqual(response.json()["context_window"], 0)
        self.assertEqual(response.json()["intro_chunks"], 1)
        self.assertEqual(response.json()["max_context_chunks"], 6)

    def test_adaptive_context_retry_runs_when_judge_rejects(self):
        settings = replace(
            Settings.from_env(),
            eager_load=False,
            yandex_api_key="test",
            yandex_folder_id="folder",
            top_k=3,
            context_window=0,
            context_intro_chunks=0,
            max_context_chunks=6,
            adaptive_top_k=5,
            adaptive_context_window=1,
            adaptive_context_intro_chunks=1,
            adaptive_max_context_chunks=8,
            enable_adaptive_context=True,
        )
        retriever = FakeRetriever()
        generator = RejectThenAcceptGenerator()
        client_context = TestClient(
            create_app(
                service=RAGService(settings, retriever=retriever, generator=generator),
                settings=settings,
            )
        )
        with client_context as client:
            response = client.post(
                "/api/v1/ask",
                json={"question": "How do I renew?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Grounded expanded answer [1].")
        self.assertTrue(body["guardrail"]["accepted"])
        self.assertTrue(body["adaptive_context"]["triggered"])
        self.assertTrue(body["adaptive_context"]["used_retry_answer"])
        self.assertEqual(body["top_k"], 5)
        self.assertEqual(body["context_window"], 1)
        self.assertEqual(body["intro_chunks"], 1)
        self.assertEqual(body["max_context_chunks"], 8)
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(generator.generate_calls, 2)
        self.assertEqual(generator.judge_calls, 2)

    def test_rejects_empty_question(self):
        response = self.client.post("/api/v1/ask", json={"question": " "})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
