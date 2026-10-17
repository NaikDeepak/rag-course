"""Day 02 — Lesson 3.2: Creating Augmented Queries Using LangChain.

Run:
    .venv/bin/python -m examples.day02_augmented_queries
"""

from __future__ import annotations

from pathlib import Path

from core.config import get_embeddings
from core.generation import build_augmented_query
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents


def run_day02_demo(
    question: str = "Why do we split documents into smaller chunks during indexing?",
) -> str:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"
    docs = load_file_documents(sample_path)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)
    vector_store = build_vector_store(chunks, get_embeddings(), collection_name="day02_demo")

    retrieved = retrieve_documents(vector_store, query=question, k=2)
    augmented_prompt = build_augmented_query(question=question, docs=retrieved)

    print("=== Day 02: Augmented Query Prompt ===")
    print(augmented_prompt)
    return augmented_prompt


if __name__ == "__main__":
    run_day02_demo()
