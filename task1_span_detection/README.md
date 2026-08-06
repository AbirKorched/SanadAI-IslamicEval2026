# Task 1: Arabic Islamic Text Span Detection

This repository contains our submission for **Task 1: Span Detection** of the shared task competition. The objective is to identify and extract meaningful semantic spans from Arabic Islamic texts using transformer-based Named Entity Recognition (NER).

## Task Description

The system detects four entity types:

| Entity | Description |
|---|---|
| `ayah` | Quranic verse spans |
| `matn` | Hadith main text |
| `isnad` | Narration chain |
| `claimed_source` | Referenced source spans |

## Approach

We fine-tune an Arabic transformer encoder for token classification using BIO tagging.

The pipeline includes:

- Transformer-based NER fine-tuning
- Class-weighted loss to handle label imbalance
- Subword-to-word prediction aggregation
- Entity-specific confidence thresholds
- Post-processing for span refinement:
  - boundary cleaning
  - span merging
  - minimum length filtering
  - source validation

## Training

Training is configured using Hydra configuration files:

```

configs/config.yaml

```

Run training with:

```bash
python scripts/train.py \
    model.name=NAMAA-Space/AraModernBert-Base-V1.0 \
    training.learning_rate=1e-5 \
    training.epochs=15 \
    training.output_dir=outputs/modernbert_lr1e5
````

The training pipeline includes:

* Dataset preparation and token alignment
* Model fine-tuning
* Evaluation using F1 score
* Saving checkpoint on every epoch

## Inference

Configure inference settings:

```
configs/inference.yaml
```

Run:

```bash
python inference.py model.checkpoint=outputs/modernbert_lr2e5/checkpoint-7275
```

The inference pipeline generates the final predictions in the required TSV format:

```
Response_ID    Annotation_ID    Segment_Type    Span_Start    Span_End    Score
```
