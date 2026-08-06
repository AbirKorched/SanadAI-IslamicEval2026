# src/dataset/parser.py

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset


@dataclass
class Span:
    entity: str
    start: int
    end: int


def _load_spans(tsv_path: str | Path, label2id: dict) -> dict[str, list[Span]]:
    """
    Load TSV annotations grouped by Response_ID.
    """

    df = pd.read_csv(tsv_path, sep="\t")

    spans = defaultdict(list)

    for _, row in df.iterrows():

        entity = str(row["Segment_Type"]).strip().lower()

        if entity == "noannotation":
            continue

        if pd.isna(row["Span_Start"]) or pd.isna(row["Span_End"]):
            continue

        try:
            start = int(row["Span_Start"])
            end = int(row["Span_End"])
        except (ValueError, TypeError):
            continue

        if f"B-{entity}" not in label2id:
            continue

        spans[str(row["Response_ID"])].append(
            Span(
                entity=entity,
                start=start,
                end=end,
            )
        )

    return spans


def parse_and_prepare_dataset(
    tokenizer,
    jsonl_path: str | Path,
    tsv_path: str | Path,
    label2id: dict,
    max_length: int = 2048,
    shuffle: bool = True,
):
    """
    Build a HuggingFace Dataset for token classification.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer.

    jsonl_path
        Input JSONL.

    tsv_path
        Span annotations.

    label2id
        BIO mapping.

    max_length
        Maximum token length.

    shuffle
        Shuffle dataset.

    Returns
    -------
    datasets.Dataset
    """

    spans_by_response = _load_spans(
        tsv_path,
        label2id,
    )

    dataset = []

    with open(jsonl_path, encoding="utf-8") as f:

        for line in f:

            sample = json.loads(line)

            response_id = str(sample["id"])

            text = sample["generated_answer"]

            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
            )

            offsets = encoding["offset_mapping"]

            labels = ["O"] * len(offsets)

            for span in spans_by_response.get(response_id, []):

                first = True

                for token_idx, (token_start, token_end) in enumerate(offsets):

                    if token_end <= span.start:
                        continue

                    if token_start >= span.end:
                        break

                    labels[token_idx] = (
                        f"B-{span.entity}"
                        if first
                        else f"I-{span.entity}"
                    )

                    first = False

            dataset.append(
                {
                    "id": response_id,
                    "text": text,
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "labels": [
                        label2id[label]
                        for label in labels
                    ],
                }
            )

    hf_dataset = Dataset.from_list(dataset)

    if shuffle:
        hf_dataset = hf_dataset.shuffle(seed=42)

    hf_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
        ],
    )

    return hf_dataset