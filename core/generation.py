"""Lessons 3.2 & 3.3: Creating Augmented Queries and Generating Context-Driven Responses."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT_TEMPLATE = """You are a precise assistant answering questions strictly from the provided context.
If the context does not contain the answer, state clearly that the answer is not in the provided documents.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """Join retrieved Document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


def build_augmented_query(question: str, docs: list[Document]) -> str:
    """Lesson 3.2: Format the prompt template with retrieved context and user question."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    context_str = format_docs(docs)
    messages = prompt.format_messages(context=context_str, question=question)
    return "\n".join(str(m.content) for m in messages)


def generate_answer(
    question: str,
    docs: list[Document],
    chat_model: BaseChatModel | None,
) -> str:
    """Lesson 3.3: Generate a grounded answer using LCEL, or return an offline summary if no LLM key is set."""
    context_str = format_docs(docs)
    if chat_model is None:
        return (
            "[Offline Mode — Add GOOGLE_API_KEY or OPENAI_API_KEY in .env for live LLM generation]\n\n"
            f"Top Retrieved Context ({len(docs)} chunks):\n{context_str}"
        )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | chat_model | StrOutputParser()
    return chain.invoke({"context": context_str, "question": question})
