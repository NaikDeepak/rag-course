from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from core.config import get_chat_model, get_gemini_api_key

# List of example documents to be used in the database
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

# Create a Chroma database from documents using Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=get_gemini_api_key(),
    task_type="retrieval_document",
)
db = Chroma.from_texts(texts=documents, embedding=embeddings)

# Perform a similarity search
results = db.similarity_search("What is machine learning?", k=5)

# Print the results
print("\nQuery: What is machine learning?")
for i, doc in enumerate(results, 1):
    print(f"\nMatch {i}:\n{doc.page_content}\n")

# Perform similarity search using similarity_score_threshold - here even when we say 6 results but score threshold as 80% , it returns only 3 as others dont pass the threshold unlike above
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 6, "score_threshold": 0.8})


results = retriever.invoke("What is machine learning?")

# Print the results
print("\nQuery: What is machine learning?")
for i, doc in enumerate(results, 1):
    print(f"\nMatch {i}:\n{doc.page_content}\n")


# Define a template for generating answers using provided context
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

# Create a custom prompt template using the defined template
custom_rag_prompt = PromptTemplate.from_template(template)
#print(custom_rag_prompt) # Print the custom prompt template

question = "What is the future of AI?"
context = retriever.invoke(question)  # Retrieve the context based on the question

# Function to display documents
def show_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Manually format the prompt template with clean text
augmented_query = custom_rag_prompt.format(context=show_docs(context), question=question)
print("Augmented Query:")
print(augmented_query)

# Invoke the LLM with the formatted prompt
llm = get_chat_model("gemini")
response = llm.invoke(augmented_query)

print("\nHelpful Answer:")
print(response.content)