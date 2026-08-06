import logging

import hydra
import torch
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from huggingface_hub import hf_hub_download

from src.utils.constants import (
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS
)

from src.inference.runner import run_inference
from src.postprocess.pipeline import postprocess


# =====================================================
# Logging
# =====================================================

install(show_locals=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    ],
)

logger = logging.getLogger(__name__)
console = Console()


# silence noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)



@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference"
)
def main(cfg: DictConfig):


    console.rule(
        "[bold cyan]Task 1 - Span Detection Inference"
    )


    # ==============================================
    # DEVICE
    # ==============================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    console.print(
        f"Using device: [green]{device}[/green]"
    )


    # ==============================================
    # LOAD TOKENIZER
    # ==============================================

    console.print(
        f"Loading tokenizer: "
        f"[green]{cfg.model.name}[/green]"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        use_fast=True
    )


    # ==============================================
    # LOAD MODEL
    # ==============================================

    console.print(
        f"Loading checkpoint: "
        f"[green]{cfg.model.checkpoint}[/green]"
    )


    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )


    model.to(device)
    model.eval()


    # ==============================================
    # LOAD DATA
    # ==============================================

    console.print(
        f"Loading input data: "
        f"[green]{cfg.data.test.jsonl}[/green]"
    )

    jsonl_path = hf_hub_download(
            repo_id=cfg.data.dataset,
            filename=cfg.data.test.jsonl,
            repo_type="dataset",
        )

    df = pd.read_json(
        jsonl_path,
        lines=True
    )


    console.print(
        f"Input examples: [bold]{len(df):,}[/bold]"
    )


    # ==============================================
    # MODEL INFERENCE
    # ==============================================

    console.rule(
        "[bold green]Running inference"
    )


    predictions = run_inference(
        df,
        model,
        tokenizer,
        device,
        LABEL2ID,
        cfg.postprocess.thresholds
    )


    console.print(
        f"Raw predictions: [bold]{len(predictions):,}[/bold]"
    )

    # Create parent directory if missing
    output_path = Path(cfg.output.path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if cfg.output.raw_path:

        predictions.to_csv(
            cfg.output.raw_path,
            sep="\t",
            index=False
        )

        console.print(
            f"Raw predictions saved to "
            f"[bold green]{cfg.output.raw_path}[/bold green]"
        )


    # ==============================================
    # POSTPROCESSING
    # ==============================================

    console.rule(
        "[bold green]Postprocessing"
    )


    submission = postprocess(
        predictions,
        cfg.postprocess
    )


    console.print(
        f"After postprocessing: "
        f"[bold]{len(submission):,}[/bold] spans"
    )


    # ==============================================
    # SORT + ANNOTATION IDS
    # ==============================================

    submission = (
        submission
        .sort_values(
            [
                "Response_ID",
                "Span_Start"
            ]
        )
        .reset_index(drop=True)
    )


    submission["Annotation_ID"] = (
        submission
        .groupby("Response_ID")
        .cumcount()
        .add(1)
    )


    console.print(
        f"Final annotations: "
        f"[bold]{len(submission):,}[/bold]"
    )


    logger.info(
        "\n".join(
            [
                f"{k}: {v}"
                for k, v in submission["Segment_Type"]
                .value_counts()
                .items()
            ]
        )
    )


    # ==============================================
    # SAVE
    # ==============================================

    console.rule(
        "[bold green]Saving"
    )

    submission.to_csv(
        cfg.output.path,
        sep="\t",
        index=False
    )



    console.print(
        f"Submission saved to "
        f"[bold green]{cfg.output.path}[/bold green]"
    )


if __name__ == "__main__":
    main()