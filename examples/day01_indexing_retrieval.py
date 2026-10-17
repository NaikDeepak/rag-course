"""Day 01 — Lesson 3.1: Indexing and Retrieving Documents with LangChain.

Run:
    .venv/bin/python -m examples.day01_indexing_retrieval
"""

from __future__ import annotations

from pathlib import Path

from core.config import get_embeddings, resolve_provider
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents


def run_day01_demo(
    query: str = "What are the two main phases of a LangChain RAG pipeline?",
    store_type: str = "chroma",
    k: int = 2,
) -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"
    provider = resolve_provider()

    print(f"=== Day 01: Indexing & Retrieving Documents (provider={provider}, store={store_type}) ===")
    raw_docs = load_file_documents(sample_path)
    chunks = split_documents(raw_docs, chunk_size=300, chunk_overlap=40)
    print(f"Loaded {len(raw_docs)} document(s) -> split into {len(chunks)} chunk(s).")

    embeddings = get_embeddings(provider)
    vector_store = build_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        store_type="faiss" if store_type == "faiss" else "chroma",
        collection_name="day01_demo",
    )

    results = retrieve_documents(vector_store, query=query, k=k)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} chunk(s):\n")
    for idx, doc in enumerate(results, start=1):
        print(f"[{idx}] {doc.page_content}\n")


if __name__ == "__main__":
    run_day01_demo()
