# Yes, this is an excellent, industry-standard chunking strategy for legal documents. In AI architecture, this pattern is formally known as Parent-Child Retrieval (or Hierarchical Retrieval).It is far superior to standard character-count splitting (RecursiveCharacterTextSplitter) because it aligns with how human lawyers analyze contracts.
import os
import re
import uuid
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "documents")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Dictionary to mimic an in-memory Parent Store
PARENT_DOCUMENTS_STORE = {}

def parse_legal_pdf_structured(pdf_path):
    """
    Reads a legal PDF and manually extracts its raw text structure.
    Adjust the regex matching rules below to fit your document style.
    """
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Merge pages into one continuous block of text
    full_text = "\n".join([page.page_content for page in pages])
    
    # Sanitize characters to avoid Ollama NaN issues
    full_text = "".join(ch for ch in full_text if ch.isprintable() or ch in "\n\r\t")
    
    # Regex to split on major sections (e.g., "SECTION 1:", "ARTICLE II", "Article 4.")
    section_pattern = r"(?=\b(?:SECTION|Section|ARTICLE|Article)\s+\d+\b|\b(?:SECTION|Section|ARTICLE|Article)\s+[I|V|X|L|C]+\b)"
    sections_raw = re.split(section_pattern, full_text)
    
    parsed_hierarchy = []
    
    # Track preamble/intro text found before the first formal section
    intro_text = sections_raw[0].strip()
    if intro_text:
        parsed_hierarchy.append({
            "type": "Intro/Preamble",
            "title": "Introduction",
            "body": intro_text
        })
        
    for sec in sections_raw[1:]:
        lines = sec.strip().split("\n")
        if not lines:
            continue
            
        section_title = lines[0].strip()
        section_body = "\n".join(lines[1:]).strip()
        
        # Regex to split sections down into individual sub-clauses (e.g., "1.1", "(a)", "a)")
        clause_pattern = r"(?=\b\d+\.\d+\b|\([a-z]\)\s|[a-z]\)\s)"
        clauses_raw = re.split(clause_pattern, section_body)
        
        clauses_list = []
        for cl in clauses_raw:
            cleaned_cl = cl.strip()
            if cleaned_cl:
                clauses_list.append(cleaned_cl)
                
        parsed_hierarchy.append({
            "type": "Section",
            "title": section_title,
            "clauses": clauses_list,
            "full_section_text": sec.strip() # The complete contextual backup
        })
        
    return parsed_hierarchy


def process_hierarchical_documents():
    """
    Iterates through legal PDFs, generates Parent-Child pairs, 
    and yields child chunks targeting ChromaDB.
    """
    child_documents = []
    filename = ""
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            filename = file
            pdf_path = os.path.join(DATA_PATH, file)
            print(f"📁 Parsing structure for: {file}")
            
            structured_data = parse_legal_pdf_structured(pdf_path)
            
            for item in structured_data:
                parent_id = str(uuid.uuid4())
                
                if item["type"] == "Intro/Preamble":
                    # Store parent
                    PARENT_DOCUMENTS_STORE[parent_id] = item["body"]
                    # Create child
                    child_doc = Document(
                        page_content=item["body"],
                        metadata={"parent_id": parent_id, "source": filename, "type": "intro"}
                    )
                    child_documents.append(child_doc)
                    
                elif item["type"] == "Section":
                    # Store full section text as parent context
                    PARENT_DOCUMENTS_STORE[parent_id] = item["full_section_text"]
                    
                    # Create distinct searchable child documents for each isolated clause
                    for idx, clause in enumerate(item["clauses"]):
                        child_doc = Document(
                            page_content=clause,
                            metadata={
                                "parent_id": parent_id,
                                "parent_section_title": item["title"],
                                "clause_index": idx,
                                "source": filename,
                                "type": "clause"
                            }
                        )
                        child_documents.append(child_doc)
                        
    print(f"🎯 Total clauses generated for vector store embedding: {len(child_documents)}")
    return child_documents


def create_vector_store(child_chunks):
    embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")
    
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        
    # Only index the child_chunks (clauses) into your database
    vectorstore = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ ChromaDB child vector index created successfully.")
    return vectorstore


if __name__ == "__main__":
    child_chunks = process_hierarchical_documents()
    if child_chunks:
        vectorstore = create_vector_store(child_chunks)
