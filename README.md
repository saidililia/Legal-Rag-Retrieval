# Legal Consultation RAG

A retrieval-augmented generation system for question answering over a corpus of Algerian legal documents, with a focus on **retrieval quality in legal-domain RAG**.

This project investigates a central problem in legal question answering: retrieving the **right authoritative passage**, rather than simply retrieving text that is lexically or semantically related to the query.

## Research Question

**How can retrieval quality be improved in a legal-domain RAG system when legal documents contain heterogeneous structures and terminology?**

The current experiment investigates a hybrid retrieval strategy combining:

**Generic Chunking + BM25 + Dense Retrieval + Reciprocal Rank Fusion (RRF)**

The next experimental stage investigates whether **structure-aware segmentation of legal documents** can further improve retrieval precision.


## Baseline Experiment

The current system serves as the **baseline retrieval configuration** for the project:

**Generic Chunking + BM25 + Dense Retrieval + Reciprocal Rank Fusion (RRF)**

A preliminary evaluation was conducted using a small legal question set and **RAGAS** to assess the quality of the resulting RAG pipeline.

The baseline achieved:

| Metric            |      Score |
| ----------------- | ---------: |
| Faithfulness      | **0.8333** |
| Answer Relevancy  | **0.6409** |
| Context Recall    | **1.0000** |
| Context Precision | **0.6417** |

These results provide the reference point against which subsequent retrieval improvements will be evaluated.

The detailed baseline findings, retrieval diagnostics, and observations are documented in:

**[`benchmarks/baseline/ragas_baseline.md`](benchmarks/baseline/ragas_baseline.md)**

> **Note:** The baseline evaluation is preliminary and is intended primarily as a reproducible snapshot of the current system rather than a statistically conclusive evaluation.

## Next Experiment: Structure-Aware Legal Segmentation

**Status: In progress**

Legal documents contain explicit hierarchical structures such as:

* Titles
* Chapters
* Articles
* Sections
* Individual legal provisions

The current system uses generic chunking and does not explicitly preserve this legal hierarchy.

The next experiment will investigate whether incorporating legal document structure into segmentation can improve retrieval quality.

The hypothesis is that structure-aware segmentation may produce more coherent retrieval units, preserve complete legal provisions, and reduce irrelevant context supplied to the language model.

The comparison will therefore be:

```text
Current:
Generic Chunking
        ↓
BM25 + Dense Retrieval
        ↓
RRF

vs.

Planned:
Structure-Aware Segmentation
        ↓
BM25 + Dense Retrieval
        ↓
RRF
```

The structure-aware approach is **not included in the current reported results**.

## Running the Experiment

1. Install the Python dependencies.
2. Ensure Ollama is running locally.
3. Pull the required embedding and language models.
4. Place the legal documents in `documents/`.
5. Run `ingest.py` to build the vector store.
6. Run `debug_retrieval.py` to inspect retrieval behavior.
7. Run `debug_rrf.py` to inspect BM25/dense/RRF rankings.
8. Run `evaluate.py` to reproduce the preliminary RAGAS evaluation.