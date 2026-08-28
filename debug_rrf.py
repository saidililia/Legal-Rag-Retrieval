from retreiver import create_hybrid_retriever, tokenize


def debug_query(query: str, k: int = 5):

    retriever = create_hybrid_retriever()

    print("=" * 90)
    print("RRF RETRIEVAL DIAGNOSTIC")
    print("=" * 90)
    print(f"\nQuery: {query}\n")

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    tokenized_query = tokenize(query)

    bm25_scores = retriever.bm25.get_scores(tokenized_query)

    bm25_ranked = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    print("\n" + "=" * 90)
    print("BM25 TOP 20")
    print("=" * 90)

    bm25_rank_map = {}

    for rank, (idx, score) in enumerate(bm25_ranked, start=1):

        bm25_rank_map[idx] = rank

        doc = retriever.chunks[idx]

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"BM25 score: {score:.6f}")
        print(f"Source: {doc.metadata.get('source')}")
        print("Content:")
        print(doc.page_content[:500].replace("\n", " "))

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------

    dense_results = retriever.vectorstore.similarity_search_with_score(
        query,
        k=20
    )

    print("\n" + "=" * 90)
    print("DENSE TOP 20")
    print("=" * 90)

    dense_rank_map = {}

    for rank, (doc, distance) in enumerate(dense_results, start=1):

        chunk_id = doc.metadata["chunk_id"]

        dense_rank_map[chunk_id] = rank

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Dense distance: {distance:.6f}")
        print(f"Source: {doc.metadata.get('source')}")
        print("Content:")
        print(doc.page_content[:500].replace("\n", " "))

    # ---------------------------------------------------------
    # RRF
    # ---------------------------------------------------------

    rrf_scores = {}

    for rank, (idx, _) in enumerate(bm25_ranked):
        chunk_id = retriever.chunks[idx].metadata["chunk_id"]

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (
            1 / (60 + rank)
        )

    for rank, (doc, _) in enumerate(dense_results):

        chunk_id = doc.metadata["chunk_id"]

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (
            1 / (60 + rank)
        )

    final_ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 90)
    print(f"FINAL RRF TOP {k}")
    print("=" * 90)

    for final_rank, (chunk_id, rrf_score) in enumerate(
        final_ranked[:k],
        start=1
    ):

        chunk = next(
            doc for doc in retriever.chunks
            if doc.metadata["chunk_id"] == chunk_id
        )

        bm25_rank = bm25_rank_map.get(chunk_id, "N/A")
        dense_rank = dense_rank_map.get(chunk_id, "N/A")

        print(f"\nFinal Rank: {final_rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"RRF score: {rrf_score:.6f}")
        print(f"BM25 rank: {bm25_rank}")
        print(f"Dense rank: {dense_rank}")
        print(f"Source: {chunk.metadata.get('source')}")
        print("Content:")
        print(chunk.page_content[:700].replace("\n", " "))


if __name__ == "__main__":

    queries = [
        "What is the official language?",
        "What is the religion of the country?"
    ]

    for query in queries:
        debug_query(query)