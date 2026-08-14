#!/usr/bin/env python3

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

from src.common.evaluation import find_best_threshold


console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    console.print(
        Panel.fit(
            "[bold]Hadith alpha / threshold search[/bold]",
            border_style="yellow",
        )
    )

    # This script intentionally expects the score dataframe produced by
    # run_hadith.py before applying the final fixed configuration.
    score_path = Path(cfg.paths.hadith_scores)

    if not score_path.exists():
        raise FileNotFoundError(
            f"{score_path} not found. Run the score-generation stage first."
        )

    import pandas as pd
    matn_df = pd.read_pickle(score_path)

    searches = [
        (
            "MATN",
            "matn_semantic_score",
            "matn_fuzzy_score",
            "Label",
        ),
        (
            "SOURCE",
            "source_semantic_score",
            "source_entity_score",
            "Label_claimed_source",
        ),
        (
            "ISNAD",
            "isnad_semantic_score",
            "isnad_entity_score",
            "Label_isnad",
        ),
    ]

    for name, semantic_col, fuzzy_col, label_col in searches:
        best, results = find_best_threshold(
            matn_df,
            semantic_col=semantic_col,
            fuzzy_col=fuzzy_col,
            label_col=label_col,
        )

        console.print(f"\n[bold]{name} BEST[/bold]")
        console.print(best)
        console.print(
            results.sort_values("f1", ascending=False).head(20).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
