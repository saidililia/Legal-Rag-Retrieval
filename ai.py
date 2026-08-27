from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retreiver import create_hybrid_retriever


# -------------------
# LLM
# -------------------
llm = ChatOllama(
    model="llama3",
    temperature=0.1,
    base_url="http://localhost:11434"
)

# -------------------
# Embeddings
# -------------------
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# -------------------
# Prompt
# -------------------
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

# -------------------
# Retriever
# -------------------
retriever = create_hybrid_retriever()


def stream_ai_response(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    for token in chain.stream({
        "context": context,
        "question": question
    }):
        yield token


if __name__ == "__main__":
    question = input("Question: ")

    for token in stream_ai_response(question):
        print(token, end="", flush=True)
