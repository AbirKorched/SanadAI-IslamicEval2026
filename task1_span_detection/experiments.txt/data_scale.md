# Data Composition Experiment

We investigated how the amount of **synthetically generated Quran- and
Hadith-based data** affects Task 1 span-detection performance.

All experiments use the same model and training configuration and are evaluated
on the **development set**.

> **Notation:** `NK` = `N × 1,000` generated samples from **each** source.
> Thus, `6K + 6K` means 6K Quran-generated + 6K Hadith-generated samples.

---

## Results

| Configuration | Total Training Samples | Dev F1 |
|---|---:|---:|
| Competition only | 4,706 | 0.8165 |
| 2K Quran + 2K Hadith + Competition | 8,706 | 0.9577 |
| 4K Quran + 4K Hadith + Competition | 12,706 | 0.7494 |
| **6K Quran + 6K Hadith + Competition** | **16,706** | **0.9641** |
| 8K Quran + 8K Hadith + Competition | 20,706 | 0.9578 |
| **6K + 6K + Competition + Post-processing** | **16,706** | **0.9661** |

### Selected Configuration

The best training-data composition was:

```text
                Training Data
                     │
        ┌────────────┼────────────┐
        │            │            │
    Competition    Quran        Hadith
     4,706         6,000         6,000
        │            │            │
        └────────────┴────────────┘
                     │
              16,706 samples
````

The **6K + 6K + Competition** configuration achieved **96.41 F1** on the
development set.

The final post-processing stage further improved performance to:

|             |        F1 |
| ----------- | --------: |
| Development | **96.61** |
| Test        | **97.35** |

The selected configuration and post-processing pipeline were used for the
final Task 1 submission.

**Augmented dataset:**
[IslamicEval2026-Task1-augmented](https://huggingface.co/datasets/AbirKorched9/IslamicEval2026-Task1-augmented)
