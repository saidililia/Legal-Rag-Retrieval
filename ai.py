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
# Vector DB
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

Answer the user's question ONLY using the provided legal context.

If the answer is not found in the context, say:
"I could not find enough legal information in the provided documents."

Always mention that laws vary by jurisdiction.

Context:
{context}

Question:
{question}
""")

chain = prompt | llm | StrOutputParser()


# -----------------------------
# Streaming RAG
# -----------------------------

def stream_ai_response(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    for chunk in chain.stream({
        "context": context,
        "question": question
    }):
        yield chunk