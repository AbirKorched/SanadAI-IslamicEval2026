import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_binary(df, true_col, pred_col, name, console=None):
    eval_df = df.dropna(subset=[true_col, pred_col]).copy()

    y_true = eval_df[true_col].str.lower().values
    y_pred = eval_df[pred_col].str.lower().values

    lines = [
        "=" * 50,
        name,
        f"Samples: {len(eval_df)}",
        "",
        "Confusion Matrix",
        str(confusion_matrix(
            y_true,
            y_pred,
            labels=["correct", "incorrect"],
        )),
        "",
        "Accuracy",
        str(accuracy_score(y_true, y_pred)),
        "",
        "F1",
        str(f1_score(y_true, y_pred, pos_label="correct")),
        "",
        "Classification Report",
        classification_report(y_true, y_pred),
    ]
    text = "\n".join(lines)

    if console is None:
        print(text)
    else:
        console.print(text)

    return {
        "name": name,
        "samples": len(eval_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label="correct"),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }


def find_best_threshold(df, semantic_col, fuzzy_col, label_col="Label"):
    data = df.copy()

    data["y_true"] = (
        data[label_col]
        .str.lower()
        .map({"correct": 1, "incorrect": 0})
    )

    data = data.dropna(subset=["y_true"])

    y_true = data["y_true"].astype(int).values
    semantic = data[semantic_col].values
    fuzzy = data[fuzzy_col].values

    best = {
        "f1": 0,
        "alpha": None,
        "threshold": None,
    }

    results = []

    # Exact grid from the original implementation.
    for alpha in __import__("numpy").arange(0, 1.01, 0.05):
        combined = alpha * semantic + (1 - alpha) * fuzzy

        for threshold in __import__("numpy").arange(0, 1.01, 0.01):
            y_pred = (combined >= threshold).astype(int)

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            results.append({
                "alpha": round(alpha, 2),
                "threshold": round(threshold, 2),
                "f1": f1,
            })

            if f1 > best["f1"]:
                best = {
                    "f1": alpha * 0 + f1,
                    "alpha": alpha,
                    "threshold": threshold,
                }

    return best, pd.DataFrame(results)
