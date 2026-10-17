"""Practice Step 3 — Lesson 3.3: Generating Context-Driven Responses with LCEL.

Goal:
Build a LangChain Expression Language (LCEL) chain (`prompt | chat_model | StrOutputParser()`)
and generate a grounded response from retrieved context.

Run to test your work:
    .venv/bin/python -m practice.step3_generate_response
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_chat_model, get_embeddings, resolve_provider
from core.generation import RAG_PROMPT_TEMPLATE, format_docs
from core.indexing import build_vector_store, load_file_documents, split_documents
from core.retrieval import retrieve_documents

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_docs.txt"


def practice_step3() -> str:
    provider = resolve_provider()
    docs = load_file_documents(SAMPLE_PATH)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=40)
    vector_store = build_vector_store(chunks, get_embeddings(provider), collection_name="practice_step3")

    question = "What limitations of standalone LLMs does RAG solve?"
    retrieved_docs = retrieve_documents(vector_store, query=question, k=2)
    context_str = format_docs(retrieved_docs)
    chat_model = get_chat_model(provider)

    if chat_model is None:
        raise RuntimeError("No active LLM configured. Ensure GOOGLE_API_KEY or OPENAI_API_KEY is set in .env.")

    # -------------------------------------------------------------------------
    # TODO 1: Create the prompt template from RAG_PROMPT_TEMPLATE
    # Hint:
    #   prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    # -------------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # -------------------------------------------------------------------------
    # TODO 2: Build the LCEL chain using the pipe `|` operator and invoke it
    # Hint:
    #   chain = prompt | chat_model | StrOutputParser()
    #   answer = chain.invoke({"context": context_str, "question": question})
    # -------------------------------------------------------------------------
    chain = prompt | chat_model | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    # --- Verification Checks ---
    assert prompt is not None, "❌ Complete TODO 1: Create ChatPromptTemplate"
    print("✅ TODO 1 Passed: Prompt template created")

    assert answer is not None and len(answer.strip()) > 0, "❌ Complete TODO 2: Build LCEL chain and invoke it"
    print(f"✅ TODO 2 Passed: Generated grounded answer using {provider.upper()}:\n")
    print(answer)
    return answer


if __name__ == "__main__":
    practice_step3()
