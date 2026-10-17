"""Practice Step 4 — Chapter 5 Challenges: Vector Store Switching (Chroma vs FAISS).

Goal:
Index the same chunks into FAISS instead of Chroma and compare retrieval results.

Run to test your work:
    .venv/bin/python -m practice.step4_vector_stores_and_pdf
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.config import get_embeddings
from core.indexing import load_file_documents, split_documents

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"


def practice_step4() -> list[Document]:
    docs = load_file_documents(SAMPLE_PATH)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)
    embeddings = get_embeddings()
    query = "How does Streamlit help with a LangChain RAG pipeline?"

    # -------------------------------------------------------------------------
    # TODO 1: Build a FAISS vector store from `chunks` and `embeddings`
    # Hint:
    #   faiss_store = FAISS.from_documents(chunks, embeddings)
    # -------------------------------------------------------------------------
    faiss_store = FAISS.from_documents(chunks, embeddings)

    # -------------------------------------------------------------------------
    # TODO 2: Retrieve the top 1 document chunk using faiss_store.as_retriever
    # Hint:
    #   retriever = faiss_store.as_retriever(search_kwargs={"k": 1})
    #   top_matches = retriever.invoke(query)
    # -------------------------------------------------------------------------
    retriever = faiss_store.as_retriever(search_kwargs={"k": 1})
    top_matches = retriever.invoke(query)

    assert faiss_store is not None, "❌ Complete TODO 1: Build FAISS vector store"
    print("✅ TODO 1 Passed: Built FAISS index")

    assert top_matches is not None and len(top_matches) == 1, "❌ Complete TODO 2: Retrieve top-1 chunk from FAISS"
    print(f"✅ TODO 2 Passed: Top FAISS match:\n{top_matches[0].page_content}")
    return top_matches


if __name__ == "__main__":
    practice_step4()
