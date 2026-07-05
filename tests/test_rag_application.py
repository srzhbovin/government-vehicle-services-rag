import sys
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from backend.main import create_app  # noqa: E402
from rag_pipeline.generator import GenerationResult  # noqa: E402
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

    def initialize(self):
        self.ready = True

    def retrieve(self, query, top_k=None):
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

    def generate(self, question, chunks, language="auto"):
        return GenerationResult(
            answer="Grounded answer [1].",
            model="fake-model",
            usage=GenerationUsage(input_tokens=10, output_tokens=4, total_tokens=14),
            elapsed_ms=20.0,
        )


class RAGApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = replace(
            Settings.from_env(),
            eager_load=False,
            yandex_api_key="test",
            yandex_folder_id="folder",
        )
        service = RAGService(
            settings,
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
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

    def test_retrieve_endpoint(self):
        response = self.client.post(
            "/api/v1/retrieve",
            json={"question": "How do I renew?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["document_id"], "doc-1")

    def test_ask_endpoint(self):
        response = self.client.post(
            "/api/v1/ask",
            json={"question": "How do I renew?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Grounded answer [1].")
        self.assertEqual(response.json()["usage"]["total_tokens"], 14)
        self.assertEqual(response.json()["prompt_strategy"], "plain")

    def test_rejects_empty_question(self):
        response = self.client.post("/api/v1/ask", json={"question": " "})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
