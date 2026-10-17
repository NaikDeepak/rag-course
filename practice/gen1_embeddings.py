from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from core.config import get_gemini_api_key

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=get_gemini_api_key(),
    task_type="retrieval_document",
)

# Generate embeddings for a list of documents
embeddings = embeddings_model.embed_documents(
    [
        "This is the Fundamentals of RAG course.",
        "Educative is an AI-powered online learning platform.",
        "There are several Generative AI courses available on Educative.",
        "I am writing this using my keyboard.",
        "JavaScript is a good programming language",
    ]
)

print(len(embeddings))  # Print the number of embeddings generated (should be equal to the number of documents)
print(len(embeddings[0]))  # Print the length of the first embedding vector