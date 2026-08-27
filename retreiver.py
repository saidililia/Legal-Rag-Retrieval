import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ingest import load_documents, split_documents, DB_PATH



# tokenizes text, Converts text into lowercase words. Removes punctuation Example: "Hello, World!" → ["hello", "world"] Used for BM25 keyword matching.
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

# This is the main retrieval engine.
class HybridRetriever:
    def __init__(self, bm25, chunks, vectorstore):
        self.bm25 = bm25
        self.chunks = chunks
        self.vectorstore = vectorstore

    def invoke(self, query: str, k: int = 5):
        scores = defaultdict(float)

        # --------------------
        # BM25
        # --------------------
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for rank, (idx, _) in enumerate(bm25_ranked):
            scores[idx] += 1 / (60 + rank)

        # --------------------
        # Vector search (Chroma)
        # --------------------
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=20
        )

        max_distance = max((score for _, score in results), default=1)

        for rank, (doc, distance) in enumerate(results):
            chunk_id = doc.metadata["chunk_id"]

            similarity = 1 - (distance / max_distance)
            scores[chunk_id] += 1 / (60 + rank)

        # --------------------
        # Final ranking
        # --------------------
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            self.chunks[idx]
            for idx, _ in ranked[:k]
        ]


def create_hybrid_retriever():
    docs = load_documents()
    chunks = split_documents(docs)

    # BM25 corpus
    tokenized_corpus = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Embeddings + Chroma
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    return HybridRetriever(bm25, chunks, vectorstore)
