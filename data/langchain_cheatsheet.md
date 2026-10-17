# LangChain RAG Quick Reference Cheatsheet

## Core Imports
- **Loaders**: `from langchain_community.document_loaders import TextLoader, PyPDFLoader`
- **Splitters**: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- **Vector Stores**: `from langchain_community.vectorstores import Chroma, FAISS`
- **Prompts & LCEL**: `from langchain_core.prompts import ChatPromptTemplate` and `from langchain_core.output_parsers import StrOutputParser`

## Recommended Hyperparameters
- **Small Technical Docs**: `chunk_size=300`, `chunk_overlap=40`, `k=2` or `k=3`
- **Long PDF Reports**: `chunk_size=600`, `chunk_overlap=100`, `k=4`
- **LLM Temperature**: `0.0` for deterministic, citation-grounded answers
