from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# LLM
# -----------------------------

llm = ChatOllama(
    model="llama3",
    temperature=0.1,
    base_url="http://localhost:11434"
)

# -----------------------------
# Embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# -----------------------------
# Vector Store
# -----------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_template("""
You are a legal AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I could not find enough legal information in the documents."

Always mention laws vary by jurisdiction.

Context:
{context}

Question:
{question}
""")

chain = prompt | llm | StrOutputParser()


# -----------------------------
# STREAMING RAG
# -----------------------------

def stream_ai_response(question: str):

    # 1. Retrieve relevant docs
    docs = retriever.invoke(question)

    # 2. Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # 3. Stream response
    for chunk in chain.stream({
        "context": context,
        "question": question
    }):
        yield chunk