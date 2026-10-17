"""Lesson 3.1 & Ch 5: Document loading, chunking, and vector store indexing (Chroma / FAISS)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

VectorStoreType = Literal["chroma", "faiss"]


def load_file_documents(file_path: str | Path) -> list[Document]:
    """Load documents from a .txt, .md, or .pdf file."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def split_documents(
    documents: list[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    store_type: VectorStoreType = "chroma",
    collection_name: str = "educative_rag",
) -> VectorStore:
    """Create an in-memory Chroma or FAISS vector store from document chunks."""
    if store_type == "faiss":
        return FAISS.from_documents(chunks, embeddings)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
