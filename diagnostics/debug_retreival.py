from retreiver import create_hybrid_retriever


retriever = create_hybrid_retriever()

query = "What is the religion of the country?"

results = retriever.invoke(query, k=5)

print("\n" + "=" * 80)
print("BASELINE RETRIEVAL DEBUG")
print("=" * 80)

print(f"\nQuery: {query}")
print(f"Retrieved documents: {len(results)}")

for i, doc in enumerate(results, 1):
    print("\n" + "-" * 80)
    print(f"RESULT #{i}")
    print("-" * 80)

    print("\nMetadata:")

    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")

    print("\nContent:")
    print(doc.page_content[:1500])