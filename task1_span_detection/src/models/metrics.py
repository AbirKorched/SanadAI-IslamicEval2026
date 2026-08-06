import numpy as np
from seqeval.metrics import classification_report

from src.utils.constants import LABEL_LIST


def compute_metrics(eval_pred):
    """
    Compute seqeval metrics for token classification.
    """

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        pred_tags = []
        label_tags = []

        for pred_id, label_id in zip(prediction, label):

            # Ignore special tokens
            if label_id == -100:
                continue

            pred_tags.append(LABEL_LIST[pred_id])
            label_tags.append(LABEL_LIST[label_id])

        true_predictions.append(pred_tags)
        true_labels.append(label_tags)

    report = classification_report(
        true_labels,
        true_predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "accuracy": report["micro avg"]["f1-score"],
    }