import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.prompting import (  # noqa: E402
    PromptStrategy,
    judge_response_format,
    parse_judge_answer,
    parse_model_answer,
    prompt_spec,
)


class PromptingTests(unittest.TestCase):
    def test_plain_answer_extracts_citations_without_structured_parsing(self):
        parsed = parse_model_answer("Renew online [2] and keep the receipt [1].", "plain")

        self.assertEqual(parsed.answer, "Renew online [2] and keep the receipt [1].")
        self.assertEqual(parsed.source, "[1], [2]")
        self.assertIsNone(parsed.parse_success)

    def test_json_answer_is_validated(self):
        raw = json.dumps(
            {"answer": "Renew online [1].", "confidence": 0.9, "source": "[1]"}
        )
        parsed = parse_model_answer(raw, PromptStrategy.JSON)

        self.assertTrue(parsed.parse_success)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.confidence, 0.9)

    def test_markdown_fence_is_counted_as_parseable_json(self):
        raw = '```json\n{"answer":"Use form MV-44 [1].","confidence":0.8,"source":"[1]"}\n```'
        parsed = parse_model_answer(raw, PromptStrategy.PYDANTIC)

        self.assertTrue(parsed.parse_success)
        self.assertTrue(parsed.schema_valid)

    def test_invalid_json_is_reported_without_losing_raw_answer(self):
        parsed = parse_model_answer("not json", PromptStrategy.JSON)

        self.assertFalse(parsed.parse_success)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.answer, "not json")
        self.assertIn("invalid_json", parsed.parse_error)

    def test_pydantic_rejects_missing_source(self):
        parsed = parse_model_answer(
            '{"answer":"Renew online.","confidence":0.9}',
            PromptStrategy.PYDANTIC,
        )

        self.assertTrue(parsed.parse_success)
        self.assertFalse(parsed.schema_valid)
        self.assertIn("schema_validation", parsed.parse_error)

    def test_structured_output_uses_strict_json_schema(self):
        spec = prompt_spec(PromptStrategy.STRUCTURED_OUTPUT)

        self.assertEqual(spec.response_format["type"], "json_schema")
        self.assertTrue(spec.response_format["strict"])
        self.assertEqual(
            set(spec.response_format["schema"]["required"]),
            {"answer", "confidence", "source"},
        )

    def test_judge_answer_is_validated(self):
        parsed = parse_judge_answer(
            json.dumps(
                {
                    "verdict": "partially_grounded",
                    "score": 0.55,
                    "reason": "The answer missed the general rule.",
                    "corrected_answer": "Use the normal replacement process [1].",
                    "source": "[1]",
                }
            )
        )

        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.verdict, "partially_grounded")
        self.assertEqual(parsed.corrected_answer, "Use the normal replacement process [1].")

    def test_judge_schema_uses_strict_json_schema(self):
        response_format = judge_response_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["strict"])
        self.assertEqual(
            set(response_format["schema"]["required"]),
            {"verdict", "score", "reason", "corrected_answer", "source"},
        )


if __name__ == "__main__":
    unittest.main()
