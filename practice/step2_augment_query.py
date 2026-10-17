"""Practice Step 2 — Lesson 3.2: Creating Augmented Queries with ChatPromptTemplate.

Goal:
Combine retrieved document chunks with the user's question inside a LangChain ChatPromptTemplate.

Run to test your work:
    .venv/bin/python -m practice.step2_augment_query
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_embeddings
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"

PROMPT_TEXT = """You are a helpful assistant that answers strictly using the provided context.

Context:
{context}

Question:
{question}

Answer:"""


def practice_step2() -> str:
    # Setup: load, split, and retrieve top-2 chunks
    docs = load_file_documents(SAMPLE_PATH)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)
    vector_store = build_vector_store(chunks, get_embeddings(), collection_name="practice_step2")
    question = "Why do we split documents into smaller chunks during indexing?"
    retrieved_docs: list[Document] = retrieve_documents(vector_store, query=question, k=2)

    # -------------------------------------------------------------------------
    # TODO 1: Join retrieved_docs into a single context string separated by "\n\n---\n\n"
    # Hint:
    #   context_str = "\n\n---\n\n".join(doc.page_content.strip() for doc in retrieved_docs)
    # -------------------------------------------------------------------------
    context_str = "\n\n---\n\n".join(doc.page_content.strip() for doc in retrieved_docs)

    # -------------------------------------------------------------------------
    # TODO 2: Create a ChatPromptTemplate from PROMPT_TEXT and format messages
    # Hint:
    #   prompt = ChatPromptTemplate.from_template(PROMPT_TEXT)
    #   messages = prompt.format_messages(context=context_str, question=question)
    #   augmented_query = "\n".join(str(m.content) for m in messages)
    # -------------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_template(PROMPT_TEXT)
    messages = prompt.format_messages(context=context_str, question=question)
    augmented_query = "\n".join(str(m.content) for m in messages)

    # --- Verification Checks ---
    assert context_str is not None and len(context_str) > 20, "❌ Complete TODO 1: Format retrieved_docs into context_str"
    print("✅ TODO 1 Passed: Formatted retrieved chunks into context string")

    assert (
        augmented_query is not None
        and question in augmented_query
        and context_str in augmented_query
    ), "❌ Complete TODO 2: Format ChatPromptTemplate with context and question"
    print("✅ TODO 2 Passed: Built augmented prompt!\n")
    print(augmented_query)
    return augmented_query


if __name__ == "__main__":
    practice_step2()
