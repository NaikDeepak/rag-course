"""Day 03 — Lesson 3.3: Generating Context-Driven Responses with LangChain.

Run:
    .venv/bin/python -m examples.day03_context_generation
"""

from __future__ import annotations

from pathlib import Path

from core.config import get_chat_model, get_embeddings, resolve_provider
from core.generation import generate_answer
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents


def run_day03_demo(
    question: str = "What limitations of standalone LLMs does RAG solve?",
) -> str:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"
    provider = resolve_provider()
    docs = load_file_documents(sample_path)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)

    vector_store = build_vector_store(chunks, get_embeddings(provider), collection_name="day03_demo")
    retrieved = retrieve_documents(vector_store, query=question, k=2)
    chat_model = get_chat_model(provider)
    answer = generate_answer(question=question, docs=retrieved, chat_model=chat_model)

    print(f"=== Day 03: Context-Driven Response (provider={provider}) ===")
    print(f"Q: {question}\n")
    print(f"A: {answer}")
    return answer


if __name__ == "__main__":
    run_day03_demo()
