# this code is used to ingest documents and create a ChromaDB vector store for retrieval-augmented generation (RAG) with the Ollama embeddings model.
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")


def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # stable IDs for hybrid retrieval
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    vectorstore.persist()

    print("✅ ChromaDB created successfully")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
