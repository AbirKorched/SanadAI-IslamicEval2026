# Hadith Encoder Comparison

## Overview

This experiment evaluates different multilingual and Arabic embedding models
for **Hadith Matn retrieval and hallucination verification** in IslamicEval
2026 Task 2.

The objective is to determine which encoder provides the most effective
representation for retrieving the correct Hadith candidate from the reference
corpus.

All encoders are evaluated under the same retrieval and verification pipeline.
The only variable changed in this experiment is the embedding model.

---

## Experimental Setup

For each encoder:

1. Hadith Matn queries are encoded using the candidate embedding model.
2. The reference Hadith corpus is encoded using the same model.
3. A FAISS inner-product index is built over the reference embeddings.
4. The top **2048** candidates are retrieved for each query.
5. Retrieved candidates are re-ranked using character-level fuzzy matching.
6. The candidate with the highest fuzzy score is selected.
7. Semantic and fuzzy scores are combined:

```text
combined_score =
    α × semantic_score
    + (1 − α) × fuzzy_score
````

8. The final prediction is:

```text
prediction = combined_score >= threshold
```

The optimal `alpha` and `threshold` are selected independently for each
encoder using a grid search maximizing F1 on the development data.

### Fixed Parameters

| Parameter           |               Value |
| ------------------- | ------------------: |
| Top-K retrieval     |                2048 |
| Retrieval index     | FAISS `IndexFlatIP` |
| Candidate selection |      Fuzzy matching |
| Optimization metric |                  F1 |
| Alpha range         |           0.00–1.00 |
| Alpha step          |                0.05 |
| Threshold range     |           0.00–1.00 |
| Threshold step      |                0.01 |

The retrieval configuration and downstream verification pipeline remain
identical across all experiments.

---

## Results

| Encoder                       |         F1 | Alpha | Threshold | dim |
| ----------------------------- | ---------: | ----: | --------: | ---: |
| **BGE-M3**                    | **0.8730** |  0.45 |      0.74 | 1024 |
| GATE-AraBERT-v1               |     0.8608 |  0.35 |      0.79 | 786 |
| E5-all-nli-triplet-Matryoshka |     0.8571 |  0.10 |      0.90 | 384 |
| Arabic-Triplet-Matryoshka-V2  |     0.8546 |  0.15 |      0.87 | 786 |
| AraModernBERT-Base-V1.0       |     0.8434 |  0.00 |      0.83 | 786 |
| GTE-multilingual-base         |     0.8374 |  0.30 |      0.79 | 786 |


The results indicate that BGE-M3 provides the strongest representation for
Hadith Matn retrieval under the evaluated retrieval and verification setup.

---

## Interpretation of the Optimal Alpha

The optimal `alpha` values also provide an indication of how much each encoder
benefits from semantic similarity versus fuzzy matching.

For BGE-M3:

```text
alpha = 0.45
```

giving:

```text
combined_score =
    0.45 × semantic_score
    + 0.55 × fuzzy_score
```

Thus, the best configuration combines both semantic and lexical evidence.

In contrast, AraModernBERT achieves its best result with:

```text
alpha = 0.00
```

meaning that, under this setup, the fuzzy score alone produced the best
validation F1.

These values should not be interpreted as intrinsic properties of the
embedding models. They are optimal values for this particular retrieval
corpus, candidate set, and development split.

---

## Conclusion

Among the evaluated encoders, **BGE-M3 was selected for the final Hadith Hallucination Detection 
pipeline** because it achieved the highest development F1.

The final configuration is:

```yaml
encoder: BAAI/bge-m3
top_k: 2048
alpha: 0.45
threshold: 0.74
```
