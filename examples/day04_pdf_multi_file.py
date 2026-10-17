"""Day 04 — Chapter 5 Challenges: Vector Store Switching (Chroma vs FAISS), PDF & Multi-File QA.

Run:
    .venv/bin/python -m examples.day04_pdf_multi_file
"""

from __future__ import annotations

from pathlib import Path

from core.config import get_embeddings
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents


def run_day04_demo() -> None:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"
    docs = load_file_documents(sample_path)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)
    embeddings = get_embeddings()

    query = "How does Streamlit help with a LangChain RAG pipeline?"
    for store_type in ("chroma", "faiss"):
        vs = build_vector_store(chunks, embeddings, store_type=store_type, collection_name=f"day04_{store_type}")
        top = retrieve_documents(vs, query=query, k=1)
        print(f"[{store_type.upper()}] Top match: {top[0].page_content}")


if __name__ == "__main__":
    run_day04_demo()
