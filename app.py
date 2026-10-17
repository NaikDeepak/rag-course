"""Streamlit Frontend for Educative RAG + LangChain Course.

Covers:
- Ch 3: Indexing, Retrieving, Augmented Queries, Context-Driven Generation
- Ch 4: Streamlit QA System Interface + LangChain Logic
- Ch 5: Vector Store Switching (Chroma vs FAISS), PDF Uploads, Multi-File QA
- Final Project: RAG Using LangChain with Google Gemini

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from core.config import get_chat_model, get_embeddings, resolve_provider
from core.generation import build_augmented_query, generate_answer
from core.indexing import VectorStoreType, build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents

st.set_page_config(
    page_title="RAG + LangChain Workbench",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG with LangChain & Streamlit — Daily Workbench")
st.caption("Course: Fundamentals of Retrieval-Augmented Generation with LangChain (Educative)")

with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    detected_provider = resolve_provider()
    provider = st.selectbox(
        "Model Provider",
        options=["auto", "gemini", "openai", "fake"],
        index=0,
        help=f"Auto-detected from .env: {detected_provider}",
    )
    active_provider = resolve_provider(provider)
    st.info(f"Active Provider: **{active_provider.upper()}**")

    store_type: VectorStoreType = st.radio(
        "Vector Store (Ch 5 Challenge 1)",
        options=["chroma", "faiss"],
        index=0,
        horizontal=True,
    )  # type: ignore[assignment]

    chunk_size = st.slider("Chunk Size", min_value=150, max_value=1000, value=350, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10)
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=3)

    st.divider()
    st.subheader("📂 Knowledge Source (Ch 5 Challenge 2 & 3)")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files (or leave empty to use sample_docs.txt)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )


def load_active_documents() -> list[Document]:
    if not uploaded_files:
        sample_path = Path(__file__).resolve().parent / "data" / "sample_docs.txt"
        return load_file_documents(sample_path)

    docs: list[Document] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded in uploaded_files:
            tmp_path = Path(tmpdir) / uploaded.name
            tmp_path.write_bytes(uploaded.getvalue())
            loaded = load_file_documents(tmp_path)
            for d in loaded:
                d.metadata["source"] = uploaded.name
            docs.extend(loaded)
    return docs


raw_docs = load_active_documents()
chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
embeddings = get_embeddings(active_provider)
vector_store = build_vector_store(
    chunks=chunks,
    embeddings=embeddings,
    store_type=store_type,
    collection_name=f"streamlit_{store_type}_{active_provider}",
)

col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("Documents Loaded", len(raw_docs))
col_stat2.metric("Indexed Chunks", len(chunks))
col_stat3.metric("Vector Store", store_type.upper())

tab_qa, tab_inspect, tab_chunks = st.tabs(
    [
        "💬 Day 03-04: RAG QA Interface",
        "🔍 Day 01-02: Retrieval & Augmented Prompt Inspector",
        "🧩 Indexed Chunks Viewer",
    ]
)

with tab_qa:
    question = st.text_input(
        "Ask a question about your documents:",
        value="What are the two main phases of a LangChain RAG pipeline?",
    )
    if st.button("Run RAG Pipeline", type="primary") and question.strip():
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            retrieved_docs = retrieve_documents(vector_store, query=question, k=top_k)
            chat_model = get_chat_model(active_provider)
            answer = generate_answer(question=question, docs=retrieved_docs, chat_model=chat_model)

        st.subheader("Generated Answer")
        st.markdown(answer)

        with st.expander(f"View {len(retrieved_docs)} Retrieved Source Chunks", expanded=True):
            for idx, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "sample_docs.txt")
                st.markdown(f"**Chunk #{idx}** (`{Path(str(source)).name}`)")
                st.code(doc.page_content, language="text")

with tab_inspect:
    inspect_query = st.text_input(
        "Test Query for Similarity Search & Prompt Augmentation:",
        value="Why do we split documents into smaller chunks?",
        key="inspect_query",
    )
    if inspect_query.strip():
        matched = retrieve_documents(vector_store, query=inspect_query, k=top_k)
        st.subheader("Step 1: Similarity Search Results (Lesson 3.1)")
        for idx, doc in enumerate(matched, start=1):
            st.markdown(f"**Match #{idx}**")
            st.info(doc.page_content)

        st.subheader("Step 2: Augmented Prompt Sent to LLM (Lesson 3.2)")
        augmented = build_augmented_query(question=inspect_query, docs=matched)
        st.code(augmented, language="markdown")

with tab_chunks:
    st.subheader(f"All {len(chunks)} Chunks in {store_type.upper()}")
    for idx, chunk in enumerate(chunks, start=1):
        st.markdown(f"**Chunk {idx}** ({len(chunk.page_content)} chars)")
        st.text(chunk.page_content)
        st.divider()
