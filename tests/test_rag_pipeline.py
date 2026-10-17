"""Automated smoke tests for the Educative RAG + LangChain pipeline (runs offline with fake embeddings)."""

from __future__ import annotations

from pathlib import Path

from core.config import get_embeddings
from core.generation import build_augmented_query, generate_answer
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"


def test_indexing_and_retrieval_chroma() -> None:
    docs = load_file_documents(SAMPLE_FILE)
    assert len(docs) == 1

    chunks = split_documents(docs, chunk_size=250, chunk_overlap=30)
    assert len(chunks) >= 2

    embeddings = get_embeddings("fake")
    vs = build_vector_store(chunks, embeddings, store_type="chroma", collection_name="test_chroma")
    retrieved = retrieve_documents(vs, query="Indexing Phase load split embed", k=2)
    assert len(retrieved) == 2
    assert any("Indexing" in d.page_content or "RAG" in d.page_content for d in retrieved)


def test_indexing_and_retrieval_faiss() -> None:
    docs = load_file_documents(SAMPLE_FILE)
    chunks = split_documents(docs, chunk_size=250, chunk_overlap=30)
    embeddings = get_embeddings("fake")
    vs = build_vector_store(chunks, embeddings, store_type="faiss")
    retrieved = retrieve_documents(vs, query="Streamlit web interface", k=1)
    assert len(retrieved) == 1


def test_augmented_query_and_offline_generation() -> None:
    docs = load_file_documents(SAMPLE_FILE)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
    vs = build_vector_store(chunks, get_embeddings("fake"), store_type="faiss")
    retrieved = retrieve_documents(vs, query="What is RAG?", k=2)

    augmented = build_augmented_query("What is RAG?", retrieved)
    assert "What is RAG?" in augmented
    assert "Context:" in augmented

    answer = generate_answer("What is RAG?", retrieved, chat_model=None)
    assert "Offline Mode" in answer
