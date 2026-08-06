import logging

import hydra
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
from huggingface_hub import hf_hub_download

from src.dataset.parser import parse_and_prepare_dataset
from src.models.metrics import compute_metrics
from src.models.trainer import WeightedLossTrainer
from src.models.weights import calculate_class_weights
from src.utils.constants import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

install(show_locals=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
        )
    ],
    force=True,
)

logger = logging.getLogger(__name__)
console = Console()

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

hf_logging.set_verbosity_warning()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig):

    console.rule("[bold cyan]Task 1 - Span Detection Training")

    logger.info(f"Loading tokenizer: {cfg.model.name}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        use_fast=True,
    )

    logger.info("Preparing training dataset...")

    jsonl_path = hf_hub_download(
        repo_id=cfg.data.dataset,
        filename=cfg.data.train.jsonl,
        repo_type="dataset",
    )

    tsv_path = hf_hub_download(
        repo_id=cfg.data.dataset,
        filename=cfg.data.train.tsv,
        repo_type="dataset",
    )

    train_dataset = parse_and_prepare_dataset(
        tokenizer=tokenizer,
        jsonl_path=jsonl_path,
        tsv_path=tsv_path,
        label2id=LABEL2ID,
        max_length=cfg.max_length,
    )

    logger.info("Preparing validation dataset...")

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

    dev_dataset = parse_and_prepare_dataset(
        tokenizer=tokenizer,
        jsonl_path=jsonl_path,
        tsv_path=tsv_path,
        label2id=LABEL2ID,
        max_length=cfg.max_length,
    )

    console.print(
        f"[green]Train examples:[/] [bold]{len(train_dataset):,}[/]    "
        f"[cyan]Validation:[/] [bold]{len(dev_dataset):,}[/]"
    )

    logger.info("Computing class weights...")

    class_weights = calculate_class_weights(
        train_dataset,
        NUM_LABELS,
    )

    logger.info("Loading model...")

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.name,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    console.print(
        f"[bold green]Training[/] "
        f"{cfg.training.epochs} epochs "
        f"[dim](lr={cfg.training.learning_rate:g})[/]"
    )

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        report_to="tensorboard",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    console.rule("[bold green]Training")

    trainer.train()

    console.rule("[bold green]Saving")

    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    console.print(
        f"\n:white_check_mark: [bold green]Model saved to[/] "
        f"[cyan]{cfg.training.output_dir}[/]"
    )


if __name__ == "__main__":
    main()