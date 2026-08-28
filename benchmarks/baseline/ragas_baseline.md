# Legal Consultation RAG

## Current Experimental Snapshot

---

### System

**Retrieval architecture:**

* Generic chunking
* BM25 lexical retrieval
* Dense retrieval
* Reciprocal Rank Fusion (RRF)

### Document Ingestion

Documents are loaded and generically chunked through `ingest.py`.

Structure-aware document segmentation is **not currently part of the evaluated system**.

Structure-aware segmentation is an **ongoing/planned improvement**.

### Retrieval

The current retrieval pipeline consists of:

* BM25 lexical retrieval
* Dense vector retrieval
* BGE-M3 embeddings via Ollama
* Chroma vector store
* Reciprocal Rank Fusion (RRF)
* Top-k retrieval

### Evaluation

**Evaluation framework:** RAGAS

**Evaluation questions:**

1. What is the official language?
2. What is the religion of the country?

---

## Current RAGAS Results

### Overall Results

| Metric            |  Score |
| ----------------- | -----: |
| Faithfulness      | 0.8333 |
| Answer Relevancy  | 0.6409 |
| Context Recall    | 1.0000 |
| Context Precision | 0.6417 |

### Per-Question Results

#### Question 1

> What is the official language?

| Metric            |  Score |
| ----------------- | -----: |
| Faithfulness      | 0.6667 |
| Answer Relevancy  | 0.6239 |
| Context Recall    | 1.0000 |
| Context Precision | 0.7000 |

#### Question 2

> What is the religion of the country?

| Metric            |  Score |
| ----------------- | -----: |
| Faithfulness      | 1.0000 |
| Answer Relevancy  | 0.6579 |
| Context Recall    | 1.0000 |
| Context Precision | 0.5833 |

---

## Baseline Finding

* The current system is capable of retrieving sufficient information for both evaluated questions.
* Context recall is perfect for the current evaluation set (`1.0000`).
* Context precision is substantially lower than context recall (`0.6417` overall), indicating the presence of irrelevant retrieved context.
* The **"religion"** query represents a useful retrieval failure case: the authoritative constitutional provision uses the wording **"religion of the State"**, while the user query uses **"religion of the country"**.
* The current results provide a baseline against which subsequent retrieval improvements can be compared.

---

## Current Limitations

* The evaluation set currently contains only two questions.
* RAGAS scores should therefore be treated as an initial diagnostic rather than a statistically robust benchmark.
* The dense retrieval contribution requires further diagnostic validation.
* Generic chunking is currently used.
* No controlled ablation has yet been performed to isolate the contribution of:

  * BM25
  * Dense retrieval
  * RRF
