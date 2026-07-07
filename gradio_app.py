"""Gradio interface for the DMV RAG backend."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv


load_dotenv()

BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("RAG_FRONTEND_TIMEOUT_SECONDS", "240"))


def backend_health() -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=8)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def format_status() -> str:
    health = backend_health()
    if health is None:
        return (
            "Backend is not available. Start it with "
            "`powershell -ExecutionPolicy Bypass -File scripts/run_backend.ps1`."
        )
    status = "OK" if health["status"] == "ok" else "DEGRADED"
    return (
        f"Status: {status}\n\n"
        f"- retrieval: `{health['retrieval_method']}`\n"
        f"- provider: `{health['llm_provider']}`\n"
        f"- generation model: `{health['generation_model']}`\n"
        f"- prompt: `{health['prompt_strategy']}`\n"
        f"- default top-k: `{health['top_k']}`\n"
        f"- context expansion: intro `{health['context_intro_chunks']}`, "
        f"window `{health['context_window']}`, max `{health['max_context_chunks']}`\n"
        f"- generation: temperature `{health['generation_temperature']}`, "
        f"top-p `{health['generation_top_p']}`\n"
        f"- LLM judge: `{health['judge_enabled']}`, min score `{health['judge_min_score']}`"
    )


def ask_backend(
    question: str,
    language_label: str,
    top_k: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    use_judge: bool,
) -> tuple[str, str, str, list[list[Any]], str]:
    question = question.strip()
    if not question:
        return "Введите вопрос.", "", "", [], format_status()

    language = {
        "Автоматически": "auto",
        "Русский": "ru",
        "English": "en",
    }[language_label]
    payload = {
        "question": question,
        "language": language,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "use_judge": bool(use_judge),
    }

    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/ask",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", "Unknown backend error")
        except ValueError:
            detail = "Backend returned a non-JSON error."
        return f"Ошибка backend: {detail}", "", "", [], format_status()
    except httpx.HTTPError as error:
        return f"Не удалось обратиться к backend: {error}", "", "", [], format_status()

    answer = result["answer"]
    meta = format_metadata(result)
    guardrail = format_guardrail(result)
    sources = format_sources(result)
    return answer, meta, guardrail, sources, format_status()


def retrieve_backend(question: str, top_k: int) -> tuple[list[list[Any]], str]:
    question = question.strip()
    if not question:
        return [], "Введите вопрос."
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/retrieve",
            json={"question": question, "top_k": int(top_k)},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPError as error:
        return [], f"Ошибка retrieval: {error}"
    return format_sources(result), f"Retrieval time: {result['retrieval_ms']:.0f} ms"


def format_metadata(result: dict[str, Any]) -> str:
    timings = result["timings"]
    usage = result.get("usage") or {}
    lines = [
        f"Retrieval: `{result['retrieval_method']}`",
        f"Model: `{result['generation_model']}`",
        f"Prompt: `{result['prompt_strategy']}`",
        f"Language: `{result['answer_language']}`",
        f"top-k: `{result['top_k']}`",
        f"temperature: `{result['generation_temperature']}`",
        f"top-p: `{result['generation_top_p']}`",
        f"Retrieval time: `{timings['retrieval_ms']:.0f} ms`",
        f"Generation time: `{timings['generation_ms']:.0f} ms`",
        f"Total time: `{timings['total_ms']:.0f} ms`",
    ]
    if timings.get("judge_ms") is not None:
        lines.append(f"Judge time: `{timings['judge_ms']:.0f} ms`")
    if result.get("confidence") is not None:
        lines.append(f"Model confidence: `{result['confidence']:.2f}`")
    if result.get("source"):
        lines.append(f"Model sources: `{result['source']}`")
    if usage.get("total_tokens") is not None:
        lines.append(f"Tokens: `{usage['total_tokens']}`")
    return "\n".join(f"- {line}" for line in lines)


def format_guardrail(result: dict[str, Any]) -> str:
    guardrail = result.get("guardrail") or {}
    if not guardrail.get("enabled"):
        return "LLM-as-a-judge выключен для этого запроса."
    if not guardrail.get("checked"):
        return f"Judge не смог проверить ответ. Причина: {guardrail.get('reason') or 'unknown'}"

    verdict = guardrail.get("verdict")
    score = guardrail.get("score")
    accepted = guardrail.get("accepted")
    corrected = guardrail.get("corrected")
    reason = guardrail.get("reason") or ""
    lines = [
        f"Verdict: `{verdict}`",
        f"Score: `{score:.2f}`" if isinstance(score, (int, float)) else "Score: `n/a`",
        f"Accepted: `{accepted}`",
        f"Corrected answer: `{corrected}`",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if result.get("original_answer"):
        lines.append("\nOriginal answer before judge correction:\n")
        lines.append(result["original_answer"])
    return "\n".join(lines)


def format_sources(result: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for source in result.get("sources", []):
        rows.append(
            [
                source["rank"],
                source.get("title") or source["document_id"],
                source["document_id"],
                source["chunk_id"],
                round(float(source["score"]), 4),
                source["text"],
            ]
        )
    return rows


EXAMPLES = [
    "What should I do if I lost my driver license?",
    "How can I renew my vehicle registration?",
    "Do I need insurance to register my vehicle?",
    "How quickly must I report a change of address to DMV?",
    "Can I transfer my registration to another vehicle?",
]


with gr.Blocks(title="DMV RAG Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # DMV RAG Assistant

        Поиск и ответы по документам MultiDoc2Dial DMV. Внутри: BM25 + FAISS,
        cross-encoder reranker, расширение контекста по чанкам, YandexGPT и
        LLM-as-a-judge для проверки groundedness.
        """
    )

    with gr.Row():
        status_box = gr.Markdown(value=format_status(), label="Статус")

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Вопрос",
                placeholder="Например: What should I do if I lost my driver license?",
                lines=4,
                value=EXAMPLES[0],
            )
            gr.Examples(EXAMPLES, inputs=question_box)
            with gr.Row():
                ask_button = gr.Button("Спросить RAG", variant="primary")
                retrieve_button = gr.Button("Только retrieval")
                refresh_button = gr.Button("Обновить статус")

        with gr.Column(scale=1):
            language_box = gr.Radio(
                ["Автоматически", "Русский", "English"],
                value="Автоматически",
                label="Язык ответа",
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=6,
                label="top-k retrieval hits",
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.1,
                label="temperature",
            )
            top_p_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
                label="top-p",
            )
            max_tokens_slider = gr.Slider(
                minimum=128,
                maximum=1500,
                step=32,
                value=700,
                label="max output tokens",
            )
            judge_checkbox = gr.Checkbox(
                value=True,
                label="Проверять ответ через LLM-as-a-judge",
            )

    answer_box = gr.Markdown(label="Ответ")

    with gr.Row():
        metadata_box = gr.Markdown(label="Метаданные")
        guardrail_box = gr.Markdown(label="Judge")

    sources_table = gr.Dataframe(
        headers=["#", "Title", "Document ID", "Chunk ID", "Score", "Text"],
        datatype=["number", "str", "str", "str", "number", "str"],
        label="Найденные фрагменты",
        wrap=True,
        interactive=False,
    )
    retrieval_status_box = gr.Markdown()

    ask_button.click(
        ask_backend,
        inputs=[
            question_box,
            language_box,
            top_k_slider,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            judge_checkbox,
        ],
        outputs=[
            answer_box,
            metadata_box,
            guardrail_box,
            sources_table,
            status_box,
        ],
    )
    retrieve_button.click(
        retrieve_backend,
        inputs=[question_box, top_k_slider],
        outputs=[sources_table, retrieval_status_box],
    )
    refresh_button.click(format_status, outputs=status_box)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
