# this retreival strategy is based on a hybrid approach that combines BM25 and vector search (Chroma) to retrieve relevant documents based on a query. The BM25 algorithm is used for traditional keyword-based retrieval, while the vector search leverages embeddings to find semantically similar documents. The final ranking of documents is determined by combining the scores from both methods.
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ..chunking.generic_chunking import load_documents, split_documents, DB_PATH


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
