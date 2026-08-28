import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Load pdf documents from the "documents" folder
def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    return documents

# Split documents into chunks of 750 characters with 150 characters overlap
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

# Create a Chroma vector store from the document chunks using bge-m3 embedding model from Ollama
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

    print("ChromaDB created successfully")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
