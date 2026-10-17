"""Practice Step 1 — Lesson 3.1: Indexing & Retrieving Documents with LangChain.

Goal:
Write the 4 core lines of the RAG Indexing & Retrieval pipeline yourself!

Run to test your work:
    .venv/bin/python -m practice.step1_index_retrieve
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_embeddings

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"


def practice_step1() -> list[Document]:
    # -------------------------------------------------------------------------
    # TODO 1: Load the text file using TextLoader
    # Hint:
    #   loader = TextLoader(str(SAMPLE_PATH), encoding="utf-8")
    #   raw_docs = loader.load()
    # -------------------------------------------------------------------------
    loader = TextLoader(str(SAMPLE_PATH), encoding="utf-8")
    raw_docs = loader.load()

    # -------------------------------------------------------------------------
    # TODO 2: Split raw_docs into smaller chunks using RecursiveCharacterTextSplitter
    # Hint:
    #   splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    #   chunks = splitter.split_documents(raw_docs)
    # -------------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(raw_docs)

    # -------------------------------------------------------------------------
    # TODO 3: Embed chunks & store them in Chroma vector database
    # Hint:
    #   embeddings = get_embeddings()
    #   vector_store = Chroma.from_documents(
    #       documents=chunks,
    #       embedding=embeddings,
    #       collection_name="practice_step1",
    #   )
    # -------------------------------------------------------------------------
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="practice_step1",
    )

    # -------------------------------------------------------------------------
    # TODO 4: Create a retriever and retrieve the top 2 chunks for the query
    # Hint:
    #   retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    #   results = retriever.invoke(query)
    # -------------------------------------------------------------------------
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    query = "What are the two main phases of a LangChain RAG pipeline?"
    results = retriever.invoke(query)

    # --- Verification Checks ---
    assert raw_docs is not None and len(raw_docs) == 1, "❌ Complete TODO 1: Load raw_docs using TextLoader"
    print(f"✅ TODO 1 Passed: Loaded {len(raw_docs)} document ({len(raw_docs[0].page_content)} chars)")

    assert chunks is not None and len(chunks) > 1, "❌ Complete TODO 2: Split raw_docs into chunks"
    print(f"✅ TODO 2 Passed: Split into {len(chunks)} overlapping chunks")

    assert vector_store is not None, "❌ Complete TODO 3: Build Chroma vector_store from chunks"
    print("✅ TODO 3 Passed: Indexed chunks into ChromaDB")

    assert results is not None and len(results) == 2, "❌ Complete TODO 4: Retrieve top-2 chunks for query"
    print(f"✅ TODO 4 Passed: Retrieved {len(results)} chunks for query: '{query}'\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Match #{i} ---\n{doc.page_content}\n")

    return results


if __name__ == "__main__":
    practice_step1()
