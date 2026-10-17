"""Provider configuration for Embeddings and LLMs (Gemini, OpenAI, or Fake fallback)."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ProviderName = Literal["gemini", "openai", "fake"]


class DeterministicHashEmbeddings(Embeddings):
    """Offline deterministic embeddings for testing indexing & retrieval without API keys."""

    def __init__(self, size: int = 1536) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self.size
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_gemini_api_key() -> str | None:
    """Return the configured Gemini / Google API key from environment (.env)."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def resolve_provider(preferred: str | None = None) -> ProviderName:
    """Resolve active provider based on preference or available environment keys."""
    choice = (preferred or os.getenv("RAG_PROVIDER", "auto")).lower()
    if choice in ("gemini", "openai", "fake"):
        return choice  # type: ignore[return-value]

    if get_gemini_api_key():
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "fake"


def get_embeddings(provider: str | None = None, task_type: str | None = None) -> Embeddings:
    """Return LangChain Embeddings instance for the active provider."""
    active = resolve_provider(provider)
    if active == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=get_gemini_api_key(),
            task_type=task_type,
        )

    if active == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    return DeterministicHashEmbeddings(size=384)


def get_chat_model(provider: str | None = None, temperature: float = 0.0) -> BaseChatModel | None:
    """Return LangChain ChatModel instance, or None if running in offline 'fake' mode."""
    active = resolve_provider(provider)
    if active == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if active == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    return None
