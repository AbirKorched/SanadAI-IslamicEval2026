import pandas as pd


def build_predictions(matn_df):
    pred_df = matn_df[
        [
            "Response_ID",
            "Annotation_ID",
            "matn_prediction",
            "isnad_prediction",
            "source_prediction",
        ]
    ].melt(
        id_vars=[
            "Response_ID",
            "Annotation_ID",
        ],
        var_name="Segment_Type",
        value_name="Label",
    )

    pred_df["Segment_Type"] = pred_df["Segment_Type"].map({
        "matn_prediction": "matn",
        "isnad_prediction": "isnad",
        "source_prediction": "claimed_source",
    })

    pred_df = pred_df.dropna(
        subset=["Label"]
    ).reset_index(drop=True)

    return pred_df
