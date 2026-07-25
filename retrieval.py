import os
from operator import itemgetter

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# ==========================
# Configuration
# ==========================

INDEX_NAME = os.getenv("INDEX_NAME")

if not INDEX_NAME:
    raise ValueError("INDEX_NAME is not found in environment variables.")


# ==========================
# Initialize Components
# ==========================

print("Initializing components...")

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=1536,
)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================
# Prompt
# ==========================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant.

Answer the user's question only from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==========================
# Helper Functions
# ==========================

def format_docs(documents):
    """Convert retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in documents)


def create_retrieval_chain():
    """Create the Retrieval-Augmented Generation pipeline."""

    return (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever
            | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(question: str):
    """Return answer for a user question."""

    chain = create_retrieval_chain()

    return chain.invoke(
        {
            "question": question
        }
    )


# ==========================
# Testing
# ==========================

if __name__ == "__main__":

    query = "Key Parameters of Memory?"

    try:
        answer = ask_question(query)

        print("\nAnswer:\n")
        print(answer)

    except Exception as e:
        print(f"Error: {e}")