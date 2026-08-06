LABEL_LIST = [
    "O",
    "B-ayah", "I-ayah",
    "B-matn", "I-matn",
    "B-isnad", "I-isnad",
    "B-claimed_source", "I-claimed_source",
]

NUM_LABELS = len(LABEL_LIST)

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 1024