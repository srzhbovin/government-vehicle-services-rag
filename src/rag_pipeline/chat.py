"""Helpers for turning short dialogue turns into standalone RAG questions."""

from __future__ import annotations

import re
from typing import Sequence

from .schemas import ChatMessage


REFERENCE_PATTERN = re.compile(
    r"\b(it|this|that|they|them|there|these|those|one|ones|he|she|his|her|its|"
    r"это|этот|эта|эти|то|там|тогда|он|она|они|его|ее|её|их|такой|такая)\b",
    flags=re.IGNORECASE,
)


def build_standalone_question(messages: Sequence[ChatMessage]) -> str:
    current = last_user_message(messages)
    previous = [message for message in messages[:-1] if message.content.strip()]
    if not previous:
        return current

    compact_current = len(current.split()) <= 8
    has_reference = bool(REFERENCE_PATTERN.search(current))
    starts_as_follow_up = current.strip().lower().startswith(
        ("and ", "also ", "what about", "а ", "и ", "а если", "ещё", "еще")
    )
    if not (compact_current or has_reference or starts_as_follow_up):
        return current

    previous_user = ""
    for message in reversed(previous):
        if message.role == "user":
            previous_user = message.content.strip()
            break
    if previous_user:
        return f"{previous_user} {current}"
    return current


def build_chat_question_for_generation(
    messages: Sequence[ChatMessage],
    standalone_question: str,
) -> str:
    current = last_user_message(messages)
    history = short_history(messages[:-1], max_messages=6)
    if not history:
        return current
    return (
        "Answer the latest user question using the retrieved DMV fragments and the "
        "conversation history. Resolve references to previous turns, but do not invent "
        "facts that are not present in the retrieved fragments.\n\n"
        f"Conversation history:\n{history}\n\n"
        f"Standalone retrieval question:\n{standalone_question}\n\n"
        f"Latest user question:\n{current}"
    )


def last_user_message(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    raise ValueError("Chat request must contain at least one user message")


def short_history(messages: Sequence[ChatMessage], max_messages: int) -> str:
    selected = list(messages)[-max_messages:]
    lines = []
    for message in selected:
        content = " ".join(message.content.split())
        if len(content) > 700:
            content = content[:697].rstrip() + "..."
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
