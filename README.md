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

## Practice Exercises

Hands-on files in `practice/`. Run any of them with `.venv/bin/python -m practice.<file_name>` (no `.py`).

### Lesson-by-lesson (Gemini, following the course code)

| File | Lesson | What you build |
| :--- | :--- | :--- |
| `gen1_embeddings.py` | Ch 3.1: Indexing | Generate document embeddings with `GoogleGenerativeAIEmbeddings` and inspect vector size |
| `gen2_chroma.py` | Ch 3.1–3.2: Retrieval & Augmented Queries | Chroma store, similarity search, `similarity_score_threshold` retriever, `PromptTemplate` augmented query sent to the LLM by hand |
| `gen3_rag_chain.py` | Ch 3.3: Generating Context-Driven Responses | Guided TODOs: set up the LLM, run each step manually, then build and invoke the RAG chain `{"context": retriever \| format_docs, "question": RunnablePassthrough()} \| prompt \| llm \| StrOutputParser()` |

### TODO-style exercises (built on `core/` helpers, with pass/fail checks)

| File | Lesson | What you build |
| :--- | :--- | :--- |
| `step1_index_retrieve.py` | Ch 3.1: Indexing & Retrieving | Load, split, embed and retrieve, the 4 core pipeline lines |
| `step2_augment_query.py` | Ch 3.2: Augmented Queries | Combine retrieved chunks and the question in a `ChatPromptTemplate` |
| `step3_generate_response.py` | Ch 3.3: Context-Driven Responses | LCEL chain `prompt \| chat_model \| StrOutputParser()` for a grounded answer |
| `step4_vector_stores_and_pdf.py` | Ch 5: Challenges | Index the same chunks into FAISS vs Chroma and compare results |
