"""Lesson 3.1: Retrieving relevant chunks from a VectorStore."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


def get_retriever(
    vector_store: VectorStore,
    k: int = 3,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """Return a configured LangChain retriever from a vector store."""
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def retrieve_documents(
    vector_store: VectorStore,
    query: str,
    k: int = 3,
    search_type: str = "similarity",
) -> list[Document]:
    """Retrieve top-k matching document chunks for a query."""
    retriever = get_retriever(vector_store, k=k, search_type=search_type)
    return retriever.invoke(query)
