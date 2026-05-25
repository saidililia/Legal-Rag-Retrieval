import sys

# Explain each import
## An integration package for the LangChain framework that allows you to connect with and control Large Language Models (LLMs) running locally via Ollama.
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_simple_legal_pipeline():
    print("⏳ Initializing local Qwen 3 model via Ollama...")
    
    # 1. Initialize the local LLM engine
    # We point to your local port and set temperature low (0.1) for legal accuracy
    llm = ChatOllama(
        model="llama3",
        temperature=0.1,
        base_url="http://localhost:11434"
    )
    
    # 2. Define a structured prompt template
    # This gives the model a professional legal persona out of the box
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal consultation assistant. Answer the user's question using clear, professional legal analysis. Always include standard legal disclaimers regarding jurisdiction."),
        ("user", "{question}")
    ])
    
    # 3. Create the Chain
    # The '|' operator is LangChain's syntax for piping data sequentially
    # Input Text ➔ Prompt Template ➔ LLM Processing ➔ String Output Parser
    chain = prompt | llm | StrOutputParser()
    
    # 4. Execute and Stream the Response
    user_question = "What is the general remedy for a breach of contract?"
    print(f"\n💬 User Question: {user_question}\n")
    print("🤖 Assistant Response: ")
    
    # We use .stream() instead of .invoke() so tokens print word-by-word in real time
    for chunk in chain.stream({"question": user_question}):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n")



# Ensures that certain code runs only when the file is executed directly, and not when it is imported as a module into another script.
if __name__ == "__main__":
    run_simple_legal_pipeline()