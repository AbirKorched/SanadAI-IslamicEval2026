#!/usr/bin/env python3

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from gliner import GLiNER
from huggingface_hub import hf_hub_download

from src.ayah.data import load_annotations, load_quran
from src.ayah.prediction import build_ayah_prediction_dataframe
from src.ayah.retrieval import build_ayah_sanity_check
from src.common.evaluation import evaluate_binary

console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    console.print(
        Panel.fit(
            "[bold]IslamicEval 2026 — Quran / Ayah Hallucination Detection Pipeline[/bold]",
            border_style="cyan",
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
    _, df_ann = load_annotations(
        jsonl_path,
        tsv_path,
    )

    quran_path = hf_hub_download(
            repo_id=cfg.data.dataset,
            filename=cfg.data.quran.db,
            repo_type="dataset",
        )
    data, df_surah = load_quran(quran_path)

    console.print(
        f"[cyan]Quran corpus:[/cyan] {len(data):,} verses / "
        f"{len(df_surah):,} surahs"
    )

    df_sanity_check = build_ayah_sanity_check(
        df_ann,
        df_surah,
    )

    ayah_labels = (
        df_ann[df_ann["Segment_Type"] == "Ayah"]
        .reset_index(drop=True)["Label"]
    )

    accuracy = (
        ayah_labels == df_sanity_check["prediction"]
    ).mean()

    console.print(f"[cyan]Ayah accuracy after typo mask:[/cyan] {accuracy:.2%}")

    ner_model = GLiNER.from_pretrained(cfg.models.ner)

    matn_df, pred_df = build_ayah_prediction_dataframe(
        df_ann,
        df_sanity_check,
        data,
        ner_model,
    )

    evaluate_binary(
        matn_df,
        "Label_claimed_source",
        "prediction_claimed_source2",
        "QURAN CLAIMED SOURCE",
        console=console,
    )

    output = Path(cfg.paths.quran_prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output, sep="\t", index=False)

    table = Table(title="Quran pipeline output")
    table.add_column("Segment")
    table.add_column("Rows")
    for segment, count in pred_df["Segment_Type"].value_counts().items():
        table.add_row(str(segment), str(count))

    console.print(table)
    console.print(f"[bold green]✓ Saved:[/bold green] {output}")


if __name__ == "__main__":
    main()
