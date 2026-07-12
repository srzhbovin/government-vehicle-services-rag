import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline.chat import (  # noqa: E402
    build_chat_question_for_generation,
    short_history,
)
from rag_pipeline.schemas import ChatMessage  # noqa: E402


class ChatPromptTests(unittest.TestCase):
    def test_first_turn_uses_plain_question(self):
        messages = [ChatMessage(role="user", content="How do I renew my license?")]

        result = build_chat_question_for_generation(
            messages,
            "How do I renew my license?",
        )

        self.assertEqual(result, "How do I renew my license?")

    def test_generation_question_contains_previous_answer(self):
        messages = [
            ChatMessage(
                role="user",
                content="I moved to a new address. What should I do?",
            ),
            ChatMessage(
                role="assistant",
                content="Report the address change to DMV within 10 days [1].",
            ),
            ChatMessage(role="user", content="Can I do it online?"),
        ]
        standalone = "I moved to a new address. What should I do? Can I do it online?"

        result = build_chat_question_for_generation(messages, standalone)

        self.assertIn("User: I moved to a new address", result)
        self.assertIn("Assistant: Report the address change", result)
        self.assertIn(f"Standalone retrieval question:\n{standalone}", result)
        self.assertIn("Latest user question:\nCan I do it online?", result)

    def test_short_history_limits_messages_and_content_length(self):
        messages = [
            ChatMessage(role="user", content="oldest message"),
            ChatMessage(role="assistant", content="old assistant message"),
            ChatMessage(role="user", content="middle message"),
            ChatMessage(role="assistant", content="middle assistant message"),
            ChatMessage(role="user", content="recent user message"),
            ChatMessage(role="assistant", content="recent assistant message"),
            ChatMessage(role="assistant", content="x" * 900),
        ]

        result = short_history(messages, max_messages=6)

        self.assertNotIn("oldest message", result)
        self.assertIn("old assistant message", result)
        self.assertIn("x" * 697 + "...", result)


if __name__ == "__main__":
    unittest.main()
