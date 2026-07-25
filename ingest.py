import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",output_dimensionality=1536
)


def ingest_pdf(file_path):
    print("Loading PDF...")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print("PDF Loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    print(f"Chunks: {len(chunks)}")


    print("Embeddings created")

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    print("Connected to Pinecone")

    vector_store.add_documents(chunks)

    print("Upload Finished")