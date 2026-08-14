import numpy as np


def apply_scores(
    matn_df,
    alpha_bge_matn,
    threshold_bge_matn,
    alpha_bge_source,
    threshold_bge_source,
    alpha_bge_isnad,
    threshold_bge_isnad,
):
    matn_df["matn_combined_score"] = (
        alpha_bge_matn * matn_df["matn_semantic_score"]
        + (1 - alpha_bge_matn) * matn_df["matn_fuzzy_score"]
    )

    matn_df["matn_prediction"] = np.where(
        matn_df["matn_combined_score"] < threshold_bge_matn,
        "incorrect",
        "correct",
    )

    matn_df["source_combined_score"] = (
        alpha_bge_source * matn_df["source_semantic_score"]
        + (1 - alpha_bge_source) * matn_df["source_entity_score"]
    )

    matn_df["source_prediction"] = np.where(
        matn_df["source_combined_score"] < threshold_bge_source,
        "incorrect",
        "correct",
    )

    matn_df["isnad_combined_score"] = (
        alpha_bge_isnad * matn_df["isnad_semantic_score"]
        + (1 - alpha_bge_isnad) * matn_df["isnad_entity_score"]
    )

    matn_df["isnad_prediction"] = np.where(
        matn_df["isnad_combined_score"] < threshold_bge_isnad,
        "incorrect",
        "correct",
    )

    return matn_df


def mask_missing_predictions(matn_df):
    matn_df.loc[
        matn_df["Label_isnad"].isna(),
        "isnad_prediction",
    ] = np.nan

    matn_df.loc[
        matn_df["Label_claimed_source"].isna(),
        "source_prediction",
    ] = np.nan

    return matn_df
