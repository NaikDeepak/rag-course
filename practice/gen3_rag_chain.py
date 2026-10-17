"""Practice Gen 3 — Lesson 3.3: Generating Context-Driven Responses with LangChain.

Continues gen2_chroma.py. There we built the augmented query by hand and called the LLM.
Here we wire the same pieces into a RAG *chain*:

    {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()

Work through the TODOs in order. After each one, run the file — the checks at the
bottom of each step tell you whether it passed before moving on.

Run:
    .venv/bin/python -m practice.gen3_rag_chain
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.config import get_chat_model, get_gemini_api_key

# =============================================================================
# SETUP (already done in gen2) — documents, vector store, retriever, prompt
# =============================================================================
documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
    "The future of AI lies in autonomous agents, multimodal reasoning, and deep learning systems that solve complex real-world problems.",
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=get_gemini_api_key(),
    task_type="retrieval_document",
)
db = Chroma.from_texts(texts=documents, embedding=embeddings, collection_name="practice_gen3")
retriever = db.as_retriever(search_kwargs={"k": 3})

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

question = "What is the future of AI?"


def format_docs(docs):
    """Join retrieved Documents into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# =============================================================================
# TODO 1: Set up the LLM (the "generator" / master chef)
# Lesson uses: llm = ChatOpenAI(model=...)
# We use Gemini via the project helper.
# Hint:
#   llm = get_chat_model("gemini")
# =============================================================================
llm = get_chat_model("gemini")


assert llm is not None, "❌ TODO 1: create the llm (check GOOGLE_API_KEY in .env)"
print("✅ TODO 1 passed: LLM ready ->", type(llm).__name__)


# =============================================================================
# TODO 2: Do it WITHOUT a chain — every step by hand
# This is what the chain will do for you in TODO 3. Fill each line:
#   passthrough_output   = RunnablePassthrough().invoke(question)
#   retriever_output     = retriever.invoke(passthrough_output)
#   context              = format_docs(retriever_output)
#   custom_prompt_output = custom_rag_prompt.invoke({"context": context, "question": passthrough_output})
#   llm_output           = llm.invoke(custom_prompt_output)
#   manual_answer        = StrOutputParser().invoke(llm_output)
# =============================================================================
passthrough_output = RunnablePassthrough().invoke(question)
retriever_output = retriever.invoke(passthrough_output)
context = format_docs(retriever_output)
custom_prompt_output = custom_rag_prompt.invoke({"context": context, "question": passthrough_output})
llm_output = llm.invoke(custom_prompt_output)
manual_answer = StrOutputParser().invoke(llm_output)

assert passthrough_output == question, "❌ TODO 2: RunnablePassthrough should return the question unchanged"
assert retriever_output, "❌ TODO 2: retriever should return a list of Documents"
assert isinstance(manual_answer, str) and manual_answer.strip(), "❌ TODO 2: manual_answer should be a string"
print(f"✅ TODO 2 passed: retrieved {len(retriever_output)} docs")
print("   Type of llm_output   :", type(llm_output).__name__, " <- an AIMessage object")
print("   manual_answer is str?:", isinstance(manual_answer, str), "     <- StrOutputParser turned it into plain text")
print("\nManual (no chain) answer:\n", manual_answer, "\n")


# =============================================================================
# TODO 3: Build the RAG chain with the pipe `|` operator
# The dict runs both branches on the SAME input (the question string):
#   - "context":  retriever finds docs, then format_docs turns them into text
#   - "question": RunnablePassthrough passes the question through untouched
# Hint:
#   rag_chain = (
#       {"context": retriever | format_docs, "question": RunnablePassthrough()}
#       | custom_rag_prompt
#       | llm
#       | StrOutputParser()
#   )
# Note: the lesson writes {"context": retriever, ...}. That works too, but the
# prompt then receives the raw list of Document objects (with metadata/ids).
# Piping into format_docs gives the LLM clean text — try both and compare!
# =============================================================================
rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}) | custom_rag_prompt | llm | StrOutputParser()

assert rag_chain is not None, "❌ TODO 3: build rag_chain"
print("✅ TODO 3 passed: chain built ->", type(rag_chain).__name__)


# =============================================================================
# TODO 4: Trigger the chain with .invoke()
# Hint:
#   response = rag_chain.invoke(question)
# =============================================================================
response = rag_chain.invoke(question)

assert isinstance(response, str) and response.strip(), "❌ TODO 4: invoke rag_chain with the question"
print("✅ TODO 4 passed\n")
print("Question:", question)
print("Chain answer:\n", response)


# =============================================================================
# BONUS (optional): ask a question the documents CAN'T answer, e.g.
#   rag_chain.invoke("Who won the 2022 football World Cup?")
# The prompt says "If you don't know, say you don't know" — does the LLM obey?
# =============================================================================
bonus_question = "Who won the 2022 football World Cup?"
bonus_response = rag_chain.invoke(bonus_question)
print("Bonus question:", bonus_question)
print("Bonus answer:\n", bonus_response)

