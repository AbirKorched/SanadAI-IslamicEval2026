from collections import Counter

import numpy as np


def calculate_class_weights(
    dataset,
    num_labels: int,
    smoothing: str = "sqrt",
    max_weight: float = 15.0,
) -> np.ndarray:
    """
    Compute normalized class weights for token classification.

    Parameters
    ----------
    dataset
        Hugging Face Dataset containing a 'labels' column.

    num_labels
        Number of labels.

    smoothing
        Weight smoothing strategy.
            - "none": inverse frequency
            - "sqrt": square-root inverse frequency

    max_weight
        Maximum allowed class weight.

    Returns
    -------
    np.ndarray
        Normalized class weights.
    """

    counts = Counter()

    for sample in dataset:
        labels = sample["labels"]

        # Torch tensor -> list
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        counts.update(labels)

    total = sum(counts.values())

    weights = np.ones(num_labels, dtype=np.float32)

    for label_id in range(num_labels):

        count = counts.get(label_id, 0)

        if count > 0:
            weights[label_id] = total / (num_labels * count)

    if smoothing == "sqrt":
        weights = np.sqrt(weights)

    elif smoothing == "none":
        pass

    else:
        raise ValueError(
            f"Unknown smoothing method: {smoothing}"
        )

    # Normalize so background ("O") has weight = 1
    weights /= weights[0]

    # Prevent extremely large weights
    weights = np.clip(
        weights,
        a_min=1.0,
        a_max=max_weight,
    )

    return weights