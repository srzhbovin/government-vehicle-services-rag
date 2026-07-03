"""Streamlit user interface for the DMV RAG backend."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="DMV RAG Assistant",
    page_icon="🚗",
    layout="wide",
)

st.title("DMV RAG Assistant")
st.caption(
    "Поиск по документам MultiDoc2Dial DMV: BM25 + FAISS, reranker и YandexGPT"
)


@st.cache_data(ttl=5, show_spinner=False)
def backend_health() -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


health = backend_health()
with st.sidebar:
    st.subheader("Состояние")
    if health is None:
        st.error("Backend недоступен")
        st.code("powershell -ExecutionPolicy Bypass -File scripts/run_backend.ps1")
    else:
        st.success("Backend подключён")
        st.write(f"Retrieval: `{health['retrieval_method']}`")
        st.write(f"Provider: `{health['llm_provider']}`")
        st.write(f"Генерация: `{health['generation_model']}`")
        if not health["generator_configured"]:
            st.warning("Не настроены переменные Yandex Cloud в .env")
    st.divider()
    st.caption(f"API: {BACKEND_URL}")

examples = [
    "How can I renew my vehicle registration?",
    "What should I do if I lost my driver license?",
    "How quickly must I report a change of address to DMV?",
]

if "question" not in st.session_state:
    st.session_state.question = examples[0]

selected_example = st.selectbox(
    "Пример вопроса",
    ["Свой вопрос", *examples],
)
if selected_example != "Свой вопрос":
    st.session_state.question = selected_example

question = st.text_area(
    "Вопрос по документам DMV",
    key="question",
    height=100,
    placeholder="Введите вопрос...",
)

language_label = st.radio(
    "Язык ответа",
    ["Автоматически", "Русский", "English"],
    horizontal=True,
)
language = {
    "Автоматически": "auto",
    "Русский": "ru",
    "English": "en",
}[language_label]

col_ask, col_clear = st.columns([1, 5])
ask_clicked = col_ask.button("Спросить", type="primary", use_container_width=True)


def clear_question() -> None:
    st.session_state.question = ""


col_clear.button("Очистить", on_click=clear_question)

if ask_clicked:
    if not question.strip():
        st.warning("Введите вопрос")
    elif health is None:
        st.error("Сначала запустите FastAPI backend")
    else:
        try:
            with st.spinner("Ищу документы и формирую ответ..."):
                response = httpx.post(
                    f"{BACKEND_URL}/api/v1/ask",
                    json={"question": question.strip(), "language": language},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as error:
            try:
                detail = error.response.json().get("detail", "Неизвестная ошибка")
            except ValueError:
                detail = "Backend вернул некорректный ответ"
            st.error(detail)
        except httpx.HTTPError as error:
            st.error(f"Не удалось обратиться к backend: {error}")
        else:
            st.subheader("Ответ")
            st.markdown(result["answer"])

            timings = result["timings"]
            metric_columns = st.columns(4)
            metric_columns[0].metric("Retrieval", f"{timings['retrieval_ms']:.0f} мс")
            metric_columns[1].metric("Генерация", f"{timings['generation_ms']:.0f} мс")
            metric_columns[2].metric("Всего", f"{timings['total_ms']:.0f} мс")
            metric_columns[3].metric("Источников", len(result["sources"]))

            st.subheader("Найденные фрагменты")
            for source in result["sources"]:
                title = source.get("title") or source["document_id"]
                with st.expander(f"[{source['rank']}] {title}"):
                    st.caption(
                        f"document_id: {source['document_id']} · "
                        f"chunk_id: {source['chunk_id']} · score: {source['score']:.4f}"
                    )
                    st.write(source["text"])
