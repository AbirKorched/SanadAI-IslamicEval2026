#!/usr/bin/env python3

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

from src.merge.predictions import merge_predictions


console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    console.print(
        Panel.fit(
            "[bold]IslamicEval 2026 — final merge[/bold]",
            border_style="green",
        )
    )

    output = Path(cfg.paths.final_prediction)
    output.parent.mkdir(parents=True, exist_ok=True)

    df_pred = merge_predictions(
        cfg.paths.quran_prediction,
        cfg.paths.hadith_prediction,
        output,
    )

    console.print(
        f"[bold green]✓ Final submission:[/bold green] {output}"
    )
    console.print(
        df_pred["Segment_Type"].value_counts().to_string()
    )


if __name__ == "__main__":
    main()
