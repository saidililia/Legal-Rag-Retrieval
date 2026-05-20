import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize FastAPI Application
app = FastAPI(title="ZBook Legal AI Assistant Backend")

# 2. Define Request Schema
class LegalQueryRequest(BaseModel):
    question: str

# 3. Initialize your LangChain Engine
# We initialize this globally at startup so the web server doesn't rebuild the chain on every request.
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.1,
    base_url="http://localhost:11434"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert legal consultation assistant. Answer the user's question using clear, professional legal analysis. Always include standard legal disclaimers regarding jurisdiction."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 4. Asynchronous Generator for Streaming Chunks
async def generate_llm_chunks(question: str):
    """
    Invokes the LangChain stream method asynchronously. 
    Yields each text chunk as it arrives from Ollama.
    """
    try:
        # wrap the synchronous chain stream inside an async loop to prevent blocking the event loop
        for chunk in chain.stream({"question": question}):
            yield chunk
            # Small yield pause to let FastAPI event loop process network packets
            await asyncio.sleep(0.01)
    except Exception as e:
        yield f"\n[Backend Error occurred processing stream: {str(e)}]"

# 5. Define the Streaming API Endpoint
@app.post("/api/v1/consult")
async def consult_legal_assistant(payload: LegalQueryRequest):
    """
    Receives a legal question via JSON body, streams the generated 
    response back chunk-by-chunk over HTTP.
    """
    # StreamingResponse expects a generator function yielding strings
    return StreamingResponse(
        generate_llm_chunks(payload.question), 
        media_type="text/plain"
    )

# Basic health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "qwen3:8b"}