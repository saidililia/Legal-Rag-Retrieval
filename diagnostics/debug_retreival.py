
from ..retrieval.hybrid_retriever import create_hybrid_retriever
retriever = create_hybrid_retriever()

query = "What is the religion of the country?"

results = retriever.invoke(query, k=5)


print("Baseline Retrieval Debugging Output")
print("-" * 50)

print(f"\nQuery: {query}")
print(f"Retrieved documents: {len(results)}")

for i, doc in enumerate(results, 1):
    print("\n" + "-" * 50)
    print(f"Result #{i}")
    print("-" * 50)

    print("\nMetadata:")

    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")

    print("\nContent:")
    print(doc.page_content[:1500])