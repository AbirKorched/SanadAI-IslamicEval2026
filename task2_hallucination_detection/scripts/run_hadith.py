#!/usr/bin/env python3

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from huggingface_hub import hf_hub_download

from src.common.evaluation import evaluate_binary
from src.hadith.data import build_matn_df, load_annotations, load_hadith
from src.hadith.prediction import build_predictions
from src.hadith.retrieval import build_gpu_index, retrieve_matn
from src.hadith.scoring import apply_scores, mask_missing_predictions
from src.hadith.verification import (
    compute_claimed_source_scores,
    compute_isnad_scores,
)

console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    console.print(
        Panel.fit(
            "[bold]IslamicEval 2026 — Hadith Hallucination Detection Pipeline[/bold]",
            border_style="magenta",
        )
    )

    jsonl_path = hf_hub_download(
        repo_id=cfg.data.dataset,
        filename=cfg.data.dev.jsonl,
        repo_type="dataset",
    )

    tsv_path = hf_hub_download(
        repo_id=cfg.data.dataset,
        filename=cfg.data.dev.tsv,
        repo_type="dataset",
    )
    _, _, main = load_annotations(
        jsonl_path,
        tsv_path,
    )

    hadith_path = hf_hub_download(
                repo_id=cfg.data.dataset,
                filename=cfg.data.hadith.db,
                repo_type="dataset",
            )
    df_hadith = load_hadith(hadith_path)

    matn_df = build_matn_df(main)

    console.print(
        f"[cyan]Hadith corpus:[/cyan] {len(df_hadith):,}"
    )
    console.print(
        f"[cyan]MATN annotations:[/cyan] {len(matn_df):,}"
    )

    model = SentenceTransformer(
        cfg.models.embedding,
        trust_remote_code=True,
    )

    hadith_embeddings_path = hf_hub_download(
                    repo_id=cfg.data.dataset,
                    filename=cfg.data.hadith.embeddings,
                    repo_type="dataset",
                )
    embeddings = np.load(hadith_embeddings_path)
    console.print(
        f"[cyan]Embeddings:[/cyan] shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}"
    )

    gpu_index = build_gpu_index(embeddings)

    results = retrieve_matn(
        matn_df["text"].tolist(),
        model,
        gpu_index,
        df_hadith,
        top_k=cfg.hadith.top_k,
        batch_size=cfg.hadith.batch_size,
    )

    for k, v in results.items():
        matn_df[k] = v

    ner_model = GLiNER.from_pretrained(cfg.models.ner)

    matn_df = compute_claimed_source_scores(
        matn_df,
        df_hadith,
        model,
        ner_model,
    )

    matn_df = compute_isnad_scores(
        matn_df,
        df_hadith,
        model,
        ner_model,
    )

    # Persist the complete score dataframe before the final fixed thresholds.
    # This is the exact data required by the original alpha/threshold search.
    score_path = Path(cfg.paths.hadith_scores)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    matn_df.to_pickle(score_path)
    console.print(
        f"[cyan]Pre-threshold score dataframe:[/cyan] {score_path}"
    )

    s = cfg.hadith.selected

    matn_df = apply_scores(
        matn_df,
        alpha_bge_matn=float(s.matn.alpha),
        threshold_bge_matn=float(s.matn.threshold),
        alpha_bge_source=float(s.source.alpha),
        threshold_bge_source=float(s.source.threshold),
        alpha_bge_isnad=float(s.isnad.alpha),
        threshold_bge_isnad=float(s.isnad.threshold),
    )

    # Evaluation occurs before masking,
    # and then again after masking missing annotations.
    evaluate_binary(
        matn_df,
        "Label_claimed_source",
        "source_prediction",
        "CLAIMED SOURCE",
        console=console,
    )
    evaluate_binary(
        matn_df,
        "Label",
        "matn_prediction",
        "MATN",
        console=console,
    )
    evaluate_binary(
        matn_df,
        "Label_isnad",
        "isnad_prediction",
        "ISNAD",
        console=console,
    )

    matn_df = mask_missing_predictions(matn_df)

    evaluate_binary(
        matn_df,
        "Label_claimed_source",
        "source_prediction",
        "CLAIMED SOURCE (masked)",
        console=console,
    )
    evaluate_binary(
        matn_df,
        "Label",
        "matn_prediction",
        "MATN (masked)",
        console=console,
    )
    evaluate_binary(
        matn_df,
        "Label_isnad",
        "isnad_prediction",
        "ISNAD (masked)",
        console=console,
    )

    pred_df = build_predictions(matn_df)

    output = Path(cfg.paths.hadith_prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output, sep="\t", index=False)

    table = Table(title="Hadith pipeline output")
    table.add_column("Segment")
    table.add_column("Rows")
    for segment, count in pred_df["Segment_Type"].value_counts().items():
        table.add_row(str(segment), str(count))

    console.print(table)
    console.print(f"[bold green]✓ Saved:[/bold green] {output}")


if __name__ == "__main__":
    main()
