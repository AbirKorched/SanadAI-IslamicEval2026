import torch
import torch.nn as nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """
    Hugging Face Trainer with support for weighted CrossEntropyLoss.
    """

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )
        else:
            self.class_weights = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100,
            )
        else:
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100,
            )

        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss