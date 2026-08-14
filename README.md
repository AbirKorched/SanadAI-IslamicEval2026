# SanadAI — IslamicEval 2026

Official submission of the **SanadAI team** to **IslamicEval 2026**.

This repository contains our solutions for the two shared-task tracks:

- **Task 1 — Arabic Islamic Text Span Detection**
- **Task 2 — Hallucination Detection for Quranic Ayahs and Hadiths**

Each task is implemented as an independent package with its own code, configuration, experiments, and documentation.

---

## Tasks

### Task 1 — Arabic Islamic Text Span Detection

The objective of Task 1 is to identify and extract meaningful spans from Arabic Islamic texts.

Our system detects four entity types:

| Entity | Description |
|---|---|
| `ayah` | Quranic verse spans |
| `matn` | Hadith main text |
| `isnad` | Narration chain |
| `claimed_source` | Referenced source spans |

Our approach is based on transformer-based token classification with Arabic
pretrained models, followed by entity-specific post-processing and span
refinement.

**Final results:**

| Dataset | Score |
|---|---:|
| Development | **96.61** |
| Test | **97.35** |

For the implementation details, training procedure, inference, and configuration:

👉 **[Task 1 README](task1_span_detection/README.md)**

---

### Task 2 — Hallucination Detection for Quranic Ayahs and Hadiths

Task 2 evaluates whether detected Quranic and Hadith spans are faithful to
their corresponding reference sources.

The system uses different verification strategies for Quranic and Hadith
content:

- **Quran / Ayah:** direct fuzzy matching against the 114-surah Quran database,
  followed by claimed-source verification.
- **Hadith:** BGE-M3 semantic retrieval with FAISS, fuzzy candidate selection,
  and separate Matn, claimed-source, and Isnad verification.

**Final results:**

| Metric | Development | Test |
|---|---:|---:|
| Overall Accuracy | **91.05** | **86.66** |
| Ayah | **97.42** | **96.89** |
| Matn | **94.56** | **88.56** |
| Claimed Source | **85.55** | **83.76** |
| Isnad | **86.67** | **77.42** |

> Scores are reported as percentages.

For the complete architecture, pipeline details, usage instructions, and
implementation:

👉 **[Task 2 README](task2_hallucination_detection/README.md)**

---

## Repository Structure

```text
.
├── task1_span_detection/
│   ├── README.md
│   ├── configs/
│   ├── src/
│   ├── scripts/
│   └── ...
│
├── task2_hallucination_detection/
│   ├── README.md
│   ├── configs/
│   ├── src/
│   ├── scripts/
│   └── ...
│
└── README.md
````

Each task can be run independently following the instructions provided in its
corresponding README.

---

## Results Summary

| Task                                 | Development |      Test |
| ------------------------------------ | ----------: | --------: |
| **Task 1 — Span Detection**          |   **96.61** | **97.35** |
| **Task 2 — Hallucination Detection** |   **91.05** | **86.66** |

---

## Acknowledgements

We sincerely thank the **IslamicEval 2026 organizers** for organizing the
shared task and for their efforts in building a challenging benchmark for
Arabic and Islamic NLP.

We are grateful for the opportunity to participate and contribute to the
development and evaluation of Arabic NLP systems in this domain.
