# Educative RAG + LangChain + Streamlit Daily Workspace

Structured workspace for **[Fundamentals of Retrieval-Augmented Generation with LangChain](https://www.educative.io/courses/rag-llm/indexing-and-retrieving-documents-with-langchain)**.

## Quick Start

```bash
cd rag-course

# 1. Configure API key (Google Gemini or OpenAI; works offline in Fake mode too)
cp .env.example .env

# 2. Run Day 01 (Lesson 3.1: Indexing & Retrieving Documents with LangChain)
.venv/bin/python -m examples.day01_indexing_retrieval

# 3. Launch Streamlit RAG Workbench (Chapters 3, 4, 5 & Final Project)
.venv/bin/streamlit run app.py

# 4. Run tests
.venv/bin/pytest
```

## Course Map -> Code Files

| Course Section | File | Description |
| :--- | :--- | :--- |
| **Ch 3.1**: Indexing and Retrieving Documents | `core/indexing.py`, `core/retrieval.py`, `examples/day01_indexing_retrieval.py` | Load `.txt`/`.pdf`, split chunks, embed into ChromaDB / FAISS, similarity search |
| **Ch 3.2**: Creating Augmented Queries | `core/generation.py`, `examples/day02_augmented_queries.py` | Inject retrieved chunks into `ChatPromptTemplate` |
| **Ch 3.3**: Generating Context-Driven Responses | `core/generation.py`, `examples/day03_context_generation.py` | LCEL pipeline (`prompt \| llm \| StrOutputParser()`) |
| **Ch 4.1–4.3**: Build a Frontend in Streamlit | `app.py` | Interactive QA interface, chunk viewer, prompt inspector |
| **Ch 5.1–5.3**: Challenges (Vector Stores, PDF, Multi-File) | `app.py`, `examples/day04_pdf_multi_file.py` | Chroma <-> FAISS toggle, `.pdf` + multi-file uploader |
| **Project**: Build a RAG Using LangChain with Google Gemini | `core/config.py`, `app.py` | Native `GoogleGenerativeAIEmbeddings` + `ChatGoogleGenerativeAI` support |
