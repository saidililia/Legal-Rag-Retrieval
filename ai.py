from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the guardrail function from your separate file
from guardrails import is_toxic

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

Answer ONLY legally related prompts.
                                                                                    
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

# -----------------------------
# STREAMING RAG WITH GUARDRAILS
# -----------------------------
def stream_ai_response(question: str):
    
    # 🛑 1. INPUT GUARDRAIL: Check user question toxicity
    if is_toxic(question, threshold=0.6):
        yield "System Warning: Your request contains inappropriate or abusive language. Processing terminated."
        return

    # 2. Retrieve relevant docs
    docs = retriever.invoke(question)

    # 3. Build context
    context = "\n\n".join(doc.page_content for doc in docs)

    # 4. Stream response
    complete_response = ""
    for chunk in chain.stream({
        "context": context,
        "question": question
    }):
        complete_response += chunk
        yield chunk

    # 🛑 5. OUTPUT GUARDRAIL: Post-generation check on full text
    if is_toxic(complete_response, threshold=0.5):
        print(f"\n⚠️ ALERT: Post-generation guardrail triggered for output!")