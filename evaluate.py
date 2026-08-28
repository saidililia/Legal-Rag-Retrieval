import os
from datasets import Dataset
from pandas import DataFrame
from ai_response import retriever, chain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Ragas configuration
local_run_config = RunConfig(
    timeout=300,      # Give local Llama 3 up to 5 minutes per judging prompt
    max_workers=1,    # STRICTLY process 1 request at a time to prevent CPU/RAM choking
    max_retries=3,     # Retry if a local generation drops
)


# 1. Test set
test_questions = [
    {
        "question": "What is the official language?",
        "ground_truth": "Arabic and Tamazight are the official languages."
    },
    {
        "question": "What is the religion of the country?",
        "ground_truth": "Islam is the religion of the country."
    }
]


# 2. feeding the test set to RAG pipeline
print("Running questions through your RAG pipeline...")

questions = []
contexts = []
answers = []
ground_truths = []

for item in test_questions:
    q = item["question"]
    print(f"Processing: '{q}'")
    
    
    docs = retriever.invoke(q)
    retrieved_chunks = [doc.page_content for doc in docs]

    context_str = "\n\n".join(retrieved_chunks)
    
    ai_answer = chain.invoke({
        "context": context_str,
        "question": q
    })
    
    # Append to lists
    questions.append(q)
    contexts.append(retrieved_chunks)
    answers.append(ai_answer)
    ground_truths.append(item["ground_truth"])

# Package into a dataset
eval_dataset = Dataset.from_dict({
    "question": questions,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": ground_truths
})


# 3. Configuring Ragas to use local LLM and embeddings
print("\n Setting up local Ollama judges...")

from ai_response import llm, embeddings

ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)


# 4. Ragas evaluation
print("Calculating Ragas metrics (this may take a minute over local LLM)...")

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
]

result = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_emb,
    run_config=local_run_config
)


# 5. View the results
print("\n Results -----------------------------------")
print(result)

print("\n Detailed Breakdown -----------------------------------")
df = result.to_pandas()

print(df)

query_col = "user_input" if "user_input" in df.columns else "question"

available_cols = [query_col, "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
# Only slice columns that actually exist to completely guarantee no KeyError
existing_cols = [c for c in available_cols if c in df.columns]

print("\nFiltered View:")
print(df[existing_cols])

print("\n Retrieved Contexts -----------------------------------")

# Ragas columns mapping check
query_col = "user_input" if "user_input" in df.columns else "question"
answer_col = "response" if "response" in df.columns else "answer"

for index, row in df.iterrows():
    # Print the question and the generated answer safely
    print(f"\n Question: {row.get(query_col, 'N/A')}")
    print(f"AI Answer: {row.get(answer_col, 'N/A')}")
    print("-" * 50)
    print("Retrieved Chunks from ChromaDB:")
    
    # Grab the contexts column safely
    contexts_list = row.get('contexts', [])
    
    if isinstance(contexts_list, list) and len(contexts_list) > 0:
        for i, chunk in enumerate(contexts_list, 1):
            print(f"\n  [Chunk {i}]:")
            print(f"  {str(chunk).strip()}")
    else:
        print("  No text chunks found in dataframe context column.")
        
    print("-" * 50)
