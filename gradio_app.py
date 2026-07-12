"""Gradio interface for the DMV RAG backend."""

from __future__ import annotations

import html
import os
import re
from typing import Any
from urllib.parse import urlparse

import gradio as gr
import httpx
from dotenv import load_dotenv


load_dotenv()

BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("RAG_FRONTEND_TIMEOUT_SECONDS", "240"))
MAX_CHAT_MESSAGES = 20
MAX_CHAT_MESSAGE_CHARS = 4000

LANGUAGE_BY_LABEL = {
    "Автоматически": "auto",
    "Русский": "ru",
    "English": "en",
}

APP_CSS = """
.rag-header {
    padding: 18px 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ecfeff 100%);
    border: 1px solid #dbeafe;
}
.rag-header h1 {
    margin: 0 0 8px 0;
}
.rag-muted {
    color: #475569;
}
.rag-pill {
    display: inline-block;
    margin: 4px 6px 4px 0;
    padding: 4px 10px;
    border-radius: 999px;
    background: #e0f2fe;
    color: #075985;
    font-size: 0.92rem;
}
.source-card {
    margin: 10px 0;
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
}
.source-card summary {
    cursor: pointer;
}
.source-meta {
    color: #64748b;
    font-size: 0.92rem;
}
.source-text {
    margin-top: 8px;
    padding: 10px;
    border-left: 4px solid #bfdbfe;
    background: #f8fafc;
    border-radius: 8px;
}
"""


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
            "### Backend недоступен\n\n"
            "Запустите приложение командой:\n\n"
            "```powershell\n"
            "powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1\n"
            "```"
        )

    status = "OK" if health["status"] == "ok" else "DEGRADED"
    return (
        f"### Статус: `{status}`\n\n"
        f"- retrieval: `{health['retrieval_method']}`\n"
        f"- embedding model: `{health['embedding_model']}`\n"
        f"- LLM provider: `{health['llm_provider']}`\n"
        f"- generation model: `{health['generation_model']}`\n"
        f"- prompt strategy: `{health['prompt_strategy']}`\n"
        f"- default context: `top_k={health['top_k']}, "
        f"intro={health['context_intro_chunks']}, "
        f"window={health['context_window']}, "
        f"max={health['max_context_chunks']}`\n"
        f"- adaptive context: `{health['adaptive_context_enabled']}` "
        f"→ `top_k={health['adaptive_top_k']}, "
        f"intro={health['adaptive_context_intro_chunks']}, "
        f"window={health['adaptive_context_window']}, "
        f"max={health['adaptive_max_context_chunks']}`\n"
        f"- refusal gate: `{health['refusal_gate_enabled']}`, "
        f"min top score `{health['refusal_min_top_score']}`\n"
        f"- generation: `temperature={health['generation_temperature']}`, "
        f"`top_p={health['generation_top_p']}`\n"
        f"- LLM-as-a-Judge: `{health['judge_enabled']}`, "
        f"min score `{health['judge_min_score']}`"
    )


def ask_backend(
    question: str,
    language_label: str,
    top_k: int,
    context_window: int,
    intro_chunks: int,
    max_context_chunks: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    use_judge: bool,
    use_adaptive_context: bool,
) -> tuple[str, str, str, str, list[list[Any]], str]:
    question = question.strip()
    if not question:
        return "Введите вопрос.", "", "", "", [], format_status()

    payload = {
        "question": question,
        "language": LANGUAGE_BY_LABEL[language_label],
        "top_k": int(top_k),
        "context_window": int(context_window),
        "intro_chunks": int(intro_chunks),
        "max_context_chunks": int(max_context_chunks),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "use_judge": bool(use_judge),
        "use_adaptive_context": bool(use_adaptive_context),
    }

    try:
        result = post_json("/api/v1/ask", payload)
    except RuntimeError as error:
        return f"Ошибка backend: {error}", "", "", "", [], format_status()

    return (
        result["answer"],
        format_metadata(result),
        format_guardrail(result),
        format_source_cards(result),
        format_sources_table(result),
        format_status(),
    )


def retrieve_backend(
    question: str,
    top_k: int,
    context_window: int,
    intro_chunks: int,
    max_context_chunks: int,
) -> tuple[list[list[Any]], str, str]:
    question = question.strip()
    if not question:
        return [], "", "Введите вопрос."

    try:
        result = post_json(
            "/api/v1/retrieve",
            {
                "question": question,
                "top_k": int(top_k),
                "context_window": int(context_window),
                "intro_chunks": int(intro_chunks),
                "max_context_chunks": int(max_context_chunks),
            },
        )
    except RuntimeError as error:
        return [], "", f"Ошибка retrieval: {error}"

    return (
        format_sources_table(result),
        format_source_cards(result),
        f"Retrieval time: `{result['retrieval_ms']:.0f} ms`",
    )


def chat_backend(
    message: str,
    display_history: list[dict[str, str]] | None,
    clean_history: list[dict[str, str]] | None,
    language_label: str,
    top_k: int,
    context_window: int,
    intro_chunks: int,
    max_context_chunks: int,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    use_judge: bool,
    use_adaptive_context: bool,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    str,
    str,
    str,
    str,
    list[list[Any]],
    str,
]:
    message = message.strip()
    display_history = list(display_history or [])
    clean_history = list(clean_history or [])
    if not message:
        return (
            display_history,
            clean_history,
            "",
            "Введите сообщение.",
            "",
            "",
            [],
            format_status(),
        )

    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in clean_history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ][-(MAX_CHAT_MESSAGES - 2) :]
    messages.append({"role": "user", "content": message})

    payload = {
        "messages": messages,
        "language": LANGUAGE_BY_LABEL[language_label],
        "top_k": int(top_k),
        "context_window": int(context_window),
        "intro_chunks": int(intro_chunks),
        "max_context_chunks": int(max_context_chunks),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "use_judge": bool(use_judge),
        "use_adaptive_context": bool(use_adaptive_context),
    }

    updated_display_history = [
        *display_history,
        {"role": "user", "content": message},
    ][-(MAX_CHAT_MESSAGES - 1) :]
    try:
        result = post_json("/api/v1/chat", payload)
    except RuntimeError as error:
        updated_display_history.append(
            {"role": "assistant", "content": f"Ошибка backend: {error}"}
        )
        updated_display_history = updated_display_history[-MAX_CHAT_MESSAGES:]
        return (
            updated_display_history,
            clean_history,
            "",
            "",
            "",
            "",
            [],
            format_status(),
        )

    updated_display_history.append(
        {
            "role": "assistant",
            "content": format_chat_answer(result),
        }
    )
    updated_display_history = updated_display_history[-MAX_CHAT_MESSAGES:]
    updated_clean_history = [
        *messages,
        {
            "role": "assistant",
            "content": shorten(result["answer"], MAX_CHAT_MESSAGE_CHARS),
        },
    ]
    metadata = format_metadata(result)
    if result.get("standalone_question"):
        metadata += (
            "\n"
            f"- Standalone chat query: `{result['standalone_question']}`\n"
            f"- History messages used: `{result.get('history_messages', 0)}`"
        )
    return (
        updated_display_history,
        updated_clean_history,
        "",
        metadata,
        format_guardrail(result),
        format_source_cards(result),
        format_sources_table(result),
        format_status(),
    )


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", "Unknown backend error")
        except ValueError:
            detail = "Backend returned a non-JSON error."
        raise RuntimeError(str(detail)) from error
    except httpx.HTTPError as error:
        raise RuntimeError(str(error)) from error


def format_metadata(result: dict[str, Any]) -> str:
    timings = result["timings"]
    usage = result.get("usage") or {}
    lines = [
        f"Retrieval: `{result['retrieval_method']}`",
        f"Model: `{result['generation_model']}`",
        f"Prompt: `{result['prompt_strategy']}`",
        f"Language: `{result['answer_language']}`",
        f"Context: `top_k={result['top_k']}, intro={result['intro_chunks']}, "
        f"window={result['context_window']}, max={result['max_context_chunks']}`",
        f"Generation params: `temperature={result['generation_temperature']}`, "
        f"`top_p={result['generation_top_p']}`",
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

    refusal = result.get("refusal") or {}
    if refusal:
        lines.append(f"Refusal gate enabled: `{refusal.get('enabled')}`")
        lines.append(f"Refused before generation: `{refusal.get('refused')}`")
        if refusal.get("top_score") is not None:
            lines.append(
                f"Refusal top score: `{refusal['top_score']:.3f}` "
                f"(threshold `{refusal.get('min_top_score')}`)"
            )
        if refusal.get("lexical_overlap") is not None:
            lines.append(
                f"Lexical overlap: `{refusal['lexical_overlap']:.3f}` "
                f"(threshold `{refusal.get('min_lexical_overlap')}`)"
            )
        if refusal.get("reason"):
            lines.append(f"Refusal reason: {refusal['reason']}")

    adaptive = result.get("adaptive_context") or {}
    if adaptive:
        lines.append(f"Adaptive context enabled: `{adaptive.get('enabled')}`")
        lines.append(f"Adaptive context triggered: `{adaptive.get('triggered')}`")
        lines.append(f"Used retry answer: `{adaptive.get('used_retry_answer')}`")
        retry = adaptive.get("retry")
        if retry:
            lines.append(
                "Retry context: "
                f"`top_k={retry['top_k']}, intro={retry['intro_chunks']}, "
                f"window={retry['context_window']}, max={retry['max_context_chunks']}`"
            )
        if adaptive.get("reason"):
            lines.append(f"Adaptive reason: {adaptive['reason']}")

    return "\n".join(f"- {line}" for line in lines)


def format_guardrail(result: dict[str, Any]) -> str:
    guardrail = result.get("guardrail") or {}
    verdict = guardrail.get("verdict")

    if verdict == "refused_before_generation":
        return (
            "### Ответ отклонён до генерации\n\n"
            f"Причина: {guardrail.get('reason') or 'retrieval did not provide enough evidence'}"
        )
    if not guardrail.get("enabled"):
        return "LLM-as-a-Judge выключен для этого запроса."
    if not guardrail.get("checked"):
        return (
            "Judge не проверял ответ.\n\n"
            f"Причина: {guardrail.get('reason') or 'unknown'}"
        )

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


def format_sources_table(result: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for source in result.get("sources", []):
        rows.append(
            [
                source["rank"],
                source.get("title") or source["document_id"],
                source["document_id"],
                source["chunk_id"],
                round(float(source["score"]), 4),
                source.get("source_url") or "",
                source["text"],
            ]
        )
    return rows


def format_source_cards(result: dict[str, Any], max_sources: int = 8) -> str:
    sources = result.get("sources") or []
    if not sources:
        return "### Источники\n\nИсточники не найдены."

    cards = [
        "### Источники ответа\n\n"
        "Ниже показаны фрагменты, которые попали в контекст модели. "
        "Раскройте карточку, чтобы увидеть полный текст чанка."
    ]
    for source in sources[:max_sources]:
        cards.append(source_card(source))
    if len(sources) > max_sources:
        cards.append(f"\nПоказаны первые {max_sources} источников из {len(sources)}.")
    return "\n\n".join(cards)


def format_chat_answer(result: dict[str, Any]) -> str:
    answer = result["answer"].strip()
    compact_sources = format_chat_sources(result)
    if not compact_sources:
        return answer
    return f"{answer}\n\n---\n\n{compact_sources}"


def format_chat_sources(result: dict[str, Any], max_sources: int = 3) -> str:
    sources = result.get("sources") or []
    if not sources:
        return ""

    cited_ranks = {
        int(value)
        for value in re.findall(r"\[(\d+)\]", str(result.get("answer") or ""))
    }
    visible_ranks = cited_ranks | {
        int(source["rank"]) for source in sources[:max_sources]
    }
    details = ["**Источники ответа:**"]
    for source in sources:
        rank = int(source["rank"])
        if rank not in visible_ranks:
            continue
        title = escape_md(source.get("title") or source["document_id"])
        score = float(source["score"])
        url = safe_source_url(source.get("source_url"))
        url_text = (
            f' · <a href="{html.escape(url, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">документ</a>'
            if url
            else ""
        )
        text = escape_md(shorten(source["text"], 520))
        details.append(
            f"<details><summary>[{rank}] {title} · score {score:.3f}{url_text}</summary>\n\n"
            f"{text}\n\n"
            f"`chunk_id: {escape_md(source['chunk_id'])}`\n\n"
            "</details>"
        )
    return "\n\n".join(details)


def source_card(source: dict[str, Any]) -> str:
    rank = int(source["rank"])
    title = html.escape(str(source.get("title") or source["document_id"]))
    document_id = html.escape(str(source["document_id"]))
    chunk_id = html.escape(str(source["chunk_id"]))
    score = float(source["score"])
    source_url = safe_source_url(source.get("source_url"))
    url_html = (
        f'<a href="{html.escape(source_url, quote=True)}" target="_blank" '
        'rel="noopener noreferrer">открыть документ</a>'
        if source_url
        else "URL отсутствует в текущем корпусе"
    )
    text = html.escape(source["text"]).replace("\n", "<br>")
    return (
        '<details class="source-card">'
        f"<summary><b>[{rank}] {title}</b> · score {score:.4f}</summary>"
        f'<div class="source-meta">'
        f"document_id: <code>{document_id}</code><br>"
        f"chunk_id: <code>{chunk_id}</code><br>"
        f"source: {url_html}"
        "</div>"
        f'<div class="source-text">{text}</div>'
        "</details>"
    )


def shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def escape_md(value: Any) -> str:
    return html.escape(str(value or ""))


def safe_source_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


EXAMPLES = [
    "What should I do if I lost my driver license?",
    "How can I renew my vehicle registration?",
    "Do I need insurance to register my vehicle?",
    "How quickly must I report a change of address to DMV?",
    "Can I transfer my registration to another vehicle?",
    "How do I cook pasta?",
]

CHAT_EXAMPLES = [
    "I moved to a new address. What should I do?",
    "How long do I have?",
    "I lost my driver license. What should I do?",
    "How much does it cost?",
]


with gr.Blocks(
    title="DMV RAG Assistant",
    theme=gr.themes.Soft(),
    css=APP_CSS,
    analytics_enabled=False,
) as demo:
    gr.Markdown(
        """
        <div class="rag-header">
            <h1>DMV RAG Assistant</h1>
            <div class="rag-muted">
                Поиск и ответы по документам DMV из MultiDoc2Dial. Пайплайн:
                hybrid retrieval, reranker, context control, Yandex AI Studio,
                refusal gate, LLM-as-a-Judge и Chat RAG.
            </div>
            <div>
                <span class="rag-pill">BM25 + FAISS</span>
                <span class="rag-pill">cross-encoder reranker</span>
                <span class="rag-pill">structured output</span>
                <span class="rag-pill">grounded judge</span>
                <span class="rag-pill">chat history</span>
            </div>
        </div>
        """
    )

    with gr.Accordion("Состояние системы", open=False):
        status_box = gr.Markdown(value=format_status())
        refresh_button = gr.Button("Обновить статус")

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Вопрос по документам DMV",
                placeholder="Например: What should I do if I lost my driver license?",
                lines=4,
                value=EXAMPLES[0],
            )
            gr.Examples(EXAMPLES, inputs=question_box)
            with gr.Row():
                ask_button = gr.Button("Спросить RAG", variant="primary")
                retrieve_button = gr.Button("Показать только retrieval")

        with gr.Column(scale=1):
            language_box = gr.Radio(
                list(LANGUAGE_BY_LABEL.keys()),
                value="Автоматически",
                label="Язык ответа",
            )
            with gr.Accordion("Параметры retrieval и контекста", open=True):
                top_k_slider = gr.Slider(1, 10, step=1, value=3, label="top_k")
                context_window_slider = gr.Slider(
                    0,
                    3,
                    step=1,
                    value=0,
                    label="context_window",
                )
                intro_chunks_slider = gr.Slider(
                    0,
                    5,
                    step=1,
                    value=0,
                    label="intro_chunks",
                )
                max_context_chunks_slider = gr.Slider(
                    3,
                    20,
                    step=1,
                    value=6,
                    label="max_context_chunks",
                )
            with gr.Accordion("Параметры генерации", open=False):
                temperature_slider = gr.Slider(
                    0.0,
                    1.0,
                    step=0.05,
                    value=0.1,
                    label="temperature",
                )
                top_p_slider = gr.Slider(0.05, 1.0, step=0.05, value=0.9, label="top_p")
                max_tokens_slider = gr.Slider(
                    128,
                    1500,
                    step=32,
                    value=700,
                    label="max output tokens",
                )
            judge_checkbox = gr.Checkbox(
                value=True,
                label="Проверять ответ через LLM-as-a-Judge",
            )
            adaptive_context_checkbox = gr.Checkbox(
                value=True,
                label="Расширять контекст, если judge отклонил первый ответ",
            )

    answer_box = gr.Markdown(label="Ответ")

    with gr.Row():
        with gr.Accordion("Диагностика ответа", open=False):
            metadata_box = gr.Markdown()
        with gr.Accordion("Проверка groundedness", open=True):
            guardrail_box = gr.Markdown()

    with gr.Accordion("Источники и найденные фрагменты", open=True):
        source_cards_box = gr.Markdown()
        sources_table = gr.Dataframe(
            headers=[
                "#",
                "Документ",
                "Document ID",
                "Chunk ID",
                "Score",
                "URL",
                "Фрагмент",
            ],
            datatype=["number", "str", "str", "str", "number", "str", "str"],
            label="Таблица retrieval",
            wrap=True,
            interactive=False,
        )
        retrieval_status_box = gr.Markdown()

    gr.Markdown("## Chat RAG")
    gr.Markdown(
        "Диалоговый режим учитывает предыдущие сообщения. "
        "Например: сначала спросите `I moved to a new address. What should I do?`, "
        "а потом коротко `How long do I have?`."
    )
    chat_box = gr.Chatbot(
        label="Диалог",
        type="messages",
        height=420,
        show_copy_button=True,
        allow_tags=False,
    )
    chat_state = gr.State([])
    with gr.Row():
        chat_input = gr.Textbox(
            label="Сообщение в Chat RAG",
            placeholder="Например: I lost my driver license. What should I do?",
            lines=2,
            max_length=MAX_CHAT_MESSAGE_CHARS,
            scale=4,
        )
        chat_button = gr.Button("Отправить", variant="primary", scale=1)
    gr.Examples(CHAT_EXAMPLES, inputs=chat_input)
    clear_chat_button = gr.Button("Очистить диалог")

    ask_button.click(
        ask_backend,
        inputs=[
            question_box,
            language_box,
            top_k_slider,
            context_window_slider,
            intro_chunks_slider,
            max_context_chunks_slider,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            judge_checkbox,
            adaptive_context_checkbox,
        ],
        outputs=[
            answer_box,
            metadata_box,
            guardrail_box,
            source_cards_box,
            sources_table,
            status_box,
        ],
    )
    retrieve_button.click(
        retrieve_backend,
        inputs=[
            question_box,
            top_k_slider,
            context_window_slider,
            intro_chunks_slider,
            max_context_chunks_slider,
        ],
        outputs=[sources_table, source_cards_box, retrieval_status_box],
    )
    chat_button.click(
        chat_backend,
        inputs=[
            chat_input,
            chat_box,
            chat_state,
            language_box,
            top_k_slider,
            context_window_slider,
            intro_chunks_slider,
            max_context_chunks_slider,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            judge_checkbox,
            adaptive_context_checkbox,
        ],
        outputs=[
            chat_box,
            chat_state,
            chat_input,
            metadata_box,
            guardrail_box,
            source_cards_box,
            sources_table,
            status_box,
        ],
    )
    chat_input.submit(
        chat_backend,
        inputs=[
            chat_input,
            chat_box,
            chat_state,
            language_box,
            top_k_slider,
            context_window_slider,
            intro_chunks_slider,
            max_context_chunks_slider,
            temperature_slider,
            top_p_slider,
            max_tokens_slider,
            judge_checkbox,
            adaptive_context_checkbox,
        ],
        outputs=[
            chat_box,
            chat_state,
            chat_input,
            metadata_box,
            guardrail_box,
            source_cards_box,
            sources_table,
            status_box,
        ],
    )
    clear_chat_button.click(
        lambda: ([], [], "", "", "", "", []),
        outputs=[
            chat_box,
            chat_state,
            chat_input,
            metadata_box,
            guardrail_box,
            source_cards_box,
            sources_table,
        ],
    )
    refresh_button.click(format_status, outputs=status_box)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
