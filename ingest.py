from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import os

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


def create_vector_store(chunks):

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    vectorstore.persist()

    print("✅ Vector database created.")


if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents(docs)

    create_vector_store(chunks)