# Retrieval Experiment — Current Observations

## Objective

The current experiment investigates retrieval quality in a legal
consultation RAG system using generic document chunking combined with
lexical and dense retrieval.

The retrieval pipeline consists of:

- BM25 lexical retrieval
- BGE-M3 dense retrieval
- Chroma vector search
- Reciprocal Rank Fusion (RRF)

Structure-aware document segmentation is intentionally excluded from
the current experiment and is being developed as a subsequent
improvement.

## Results

The current RAGAS evaluation produced:

| Metric | Score |
|---|---:|
| Faithfulness | 0.8333 |
| Answer Relevancy | 0.6409 |
| Context Recall | 1.0000 |
| Context Precision | 0.6417 |

The strongest result is context recall (1.0), indicating that the
necessary evidence was retrieved for both evaluation questions.

Context precision (0.6417) is lower, indicating that the retrieved
context still contains irrelevant or redundant material.

## Retrieval Behavior

The retrieval diagnostics show that BM25 is highly effective for
questions whose wording overlaps directly with the source documents.
For example, the query concerning Algeria's official language retrieves
the Constitution chunk containing Article 3 at the top of the BM25
ranking.

However, lexical retrieval can also rank semantically related but
non-answering chunks highly.

The hybrid retrieval architecture is intended to mitigate this by
combining lexical and semantic rankings through RRF.

The current diagnostics suggest that BM25 has a strong influence on the
final ranking. The contribution of dense retrieval requires further
investigation before drawing conclusions about the effectiveness of RRF
itself.

## Interpretation

The current system demonstrates strong retrieval recall but only
moderate retrieval precision.

This indicates that the primary retrieval challenge is no longer simply
finding relevant evidence. The next challenge is ranking the most
useful evidence higher and reducing irrelevant context.

This is particularly important in legal consultation systems, where
retrieving the correct legal provision is more important than simply
retrieving documents containing related terminology.

## Limitations

The evaluation currently uses only two questions and therefore should
not be interpreted as a statistically robust benchmark.

The current results are best treated as an experimental snapshot that
establishes a reproducible baseline for subsequent retrieval
experiments.

## Planned Work

Structure-aware document segmentation is currently in progress.

The planned experiment will investigate whether preserving legal
document structure improves retrieval quality compared with generic
chunking.

The retrieval architecture will remain as controlled as possible so
that changes in retrieval performance can be attributed primarily to
the segmentation strategy.