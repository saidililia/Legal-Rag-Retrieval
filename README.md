# Legal Consultation RAG

A retrieval-augmented generation (RAG) system for answering questions over a corpus of Algerian legal documents.

The project investigates how retrieval strategy affects the quality and relevance of legal information supplied to a language model.

## Overview

Legal documents present a challenging retrieval setting: the answer to a question may be contained in an authoritative legal provision, while other documents contain semantically or lexically related but irrelevant information.

This project therefore focuses on improving the **retrieval layer** of a legal RAG system.

The current retrieval pipeline combines:

```text
                    User Query
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
           BM25              Dense Retrieval
        (lexical)              (BGE-M3)
              │                     │
              └──────────┬──────────┘
                         ▼
              Reciprocal Rank Fusion
                         │
                         ▼
                  Top-k Documents
                         │
                         ▼
                    LLM Answer
```

## Current System

The current snapshot uses:

* **Generic document chunking**
* **BM25** lexical retrieval
* **BGE-M3** dense retrieval
* **Reciprocal Rank Fusion (RRF)**
* **ChromaDB** for vector storage
* **Ollama** for local model inference
* **RAGAS** for evaluation

The current implementation intentionally does **not** use structure-aware legal segmentation.

## Why Hybrid Retrieval?

Legal questions can exhibit both lexical and semantic variation.

For example, a user may ask:

> "What is the religion of the country?"

while the authoritative constitutional provision states:

> "Islam is the religion of the State."

A lexical retriever such as BM25 can be affected by this vocabulary mismatch. Dense retrieval provides a complementary semantic signal, while RRF combines the rankings produced by the two retrieval methods.

The purpose of the hybrid approach is therefore to combine:

**lexical precision + semantic matching**

without requiring BM25 and dense-retrieval scores to be directly calibrated to the same scale.

## Current Retrieval Findings

Initial retrieval diagnostics reveal an important difference between the two evaluated queries.

For:

**"What is the official language?"**

BM25 ranked the relevant constitutional passage first. The retrieved chunk contains Article 3, which explicitly states that Arabic is the national and official language.

For:

**"What is the religion of the country?"**

BM25 initially ranked several unrelated chunks above the authoritative constitutional provision. The relevant constitutional passage contains Article 2:

> "Islam is the religion of the State."

This provides a concrete example of the limitations of purely lexical retrieval when the wording of a query differs from the wording used in the source document.

The experiment therefore motivates the use of hybrid lexical + dense retrieval.

## Current RAGAS Snapshot

The current evaluation uses two factual questions.

| Metric            |      Score |
| ----------------- | ---------: |
| Faithfulness      | **0.8333** |
| Answer Relevancy  | **0.6409** |
| Context Recall    | **1.0000** |
| Context Precision | **0.6417** |

Per-question results:

| Question                | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
| ----------------------- | -----------: | ---------------: | -------------: | ----------------: |
| Official language       |       0.6667 |           0.6239 |         1.0000 |            0.7000 |
| Religion of the country |       1.0000 |           0.6579 |         1.0000 |            0.5833 |

The most notable result is the difference between **context recall** and **context precision**.

The system retrieved sufficient information for both evaluated questions, but the retrieved context was not always highly focused.

This suggests that future work should focus on improving retrieval precision and document representation.

> **Caveat:** the current evaluation contains only two questions and should therefore be treated as an engineering baseline rather than a statistically significant benchmark.

## Research Direction

The current experiment is deliberately limited to:

```text
Generic Chunking
        +
BM25
        +
Dense Retrieval
        +
RRF
```

### Structure-Aware Segmentation — In Progress

A structure-aware document segmentation strategy is currently being investigated.

Legal documents contain meaningful hierarchical structures such as:

* Titles
* Chapters
* Articles
* Sections
* Legal provisions

The planned approach will preserve these structures during document segmentation instead of treating legal documents as generic sequences of text.

The objective is to investigate whether structure-aware segmentation can:

1. improve retrieval precision,
2. reduce irrelevant context,
3. preserve complete legal provisions,
4. improve the quality of context supplied to the language model.

This work is **not included in the current benchmark**. It is being treated as the next experimental stage.

## Project Structure

```text
Lawyer/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── ai.py
├── ingest.py
├── retreiver.py
├── evaluate.py
│
├── debug_retrieval.py
├── debug_rrf.py
│
├── documents/
│   └── README.md
│
├── evaluation/
│   ├── ragas_baseline.txt
│   └── retrieval_diagnostics.txt
│
└── docs/
    ├── architecture.md
    ├── benchmark.md
    └── future_work.md
```

## Reproducibility

The retrieval experiments can be reproduced by:

1. Preparing the legal document corpus.
2. Running the ingestion pipeline to create the vector store.
3. Running the retrieval diagnostics.
4. Running the RAGAS evaluation.

The local environment uses Ollama for embedding and language-model inference.

## Limitations

The current project is an early experimental snapshot. Important limitations include:

* Small evaluation set.
* Heterogeneous legal-document corpus.
* Generic rather than structure-aware chunking.
* Further investigation of dense-retrieval contribution is required.
* RRF diagnostics are still being refined.
* Current benchmark results are intended as a baseline for future experiments.

## Future Work

Planned improvements include:

* Structure-aware legal document segmentation
* Expanded retrieval evaluation dataset
* More systematic retrieval metrics such as Recall@k and Precision@k
* Improved RRF diagnostics
* Comparison of generic and structure-aware chunking
* Evaluation across different types of legal questions
* Analysis of retrieval failures and false positives

## Status

**Current:** Generic Chunking + BM25 + Dense Retrieval + RRF

**In progress:** Structure-aware legal document segmentation

**Next experimental question:**

> Does preserving the hierarchical structure of legal documents improve retrieval precision compared with generic chunking?
