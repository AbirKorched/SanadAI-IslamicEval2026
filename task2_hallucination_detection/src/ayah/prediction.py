import pandas as pd

from src.common.normalization import normalize_surah
from src.ayah.verification import extract_quran_source, evaluate_claimed_source, binary_source_score


def build_ayah_prediction_dataframe(df_ann, df_sanity_check, data, ner_model):
    keys = ["Response_ID", "Annotation_ID"]

    main = df_ann[
        df_ann["Segment_Type"].isin(["Ayah", "matn"])
    ].copy()

    claimed = (
        df_ann[df_ann["Segment_Type"] == "claimed_source"][
            keys + [
                "Segment_Type",
                "Label",
                "Span_Start",
                "Span_End",
                "text",
            ]
        ]
        .rename(
            columns={
                "Segment_Type": "Segment_Type_claimed_source",
                "Label": "Label_claimed_source",
                "Span_Start": "Span_Start_claimed_source",
                "Span_End": "Span_End_claimed_source",
                "text": "text_claimed_source",
            }
        )
    )

    main = main.merge(
        claimed,
        on=keys,
        how="left",
    )

    matn_df = (
        main[main["Segment_Type"] == "Ayah"][
            [
                "Label",
                "text",
                "Response_ID",
                "Annotation_ID",
                "text_claimed_source",
                "Label_claimed_source",
            ]
        ]
        .reset_index(drop=True)
    )

    matn_df["ayah_predition"] = df_sanity_check["prediction"]
    matn_df["surah_ids"] = df_sanity_check["matched_surah_ids"]
    matn_df["matched_num_ayahs"] = df_sanity_check["matched_num_ayahs"]

    matn_df["surahs_name"] = df_sanity_check[
        "fuzzy_search"
    ].apply(
        lambda lst: [
            normalize_surah(x)
            .replace("سوره", "")
            .strip()
            for x in lst
        ]
    )

    matn_df["text_claimed_source"] = matn_df[
        "text_claimed_source"
    ].apply(
        lambda x: normalize_surah(x)
        if pd.notna(x)
        else x
    )

    matn_df["source_entities"] = matn_df[
        "text_claimed_source"
    ].apply(
        lambda x: extract_quran_source(x, ner_model)
    )

    matn_df["prediction_claimed_source2"] = matn_df.apply(
        lambda r: evaluate_claimed_source(
            r,
            surahs=data,
        ),
        axis=1,
    )

    # Exact original binary source diagnostic, retained as a separate column.
    matn_df["source_prediction"] = matn_df.apply(
        binary_source_score,
        axis=1,
    )

    # Original code explicitly removes predictions where no source annotation exists.
    matn_df.loc[
        matn_df["Label_claimed_source"].isna(),
        "prediction_claimed_source2",
    ] = pd.NA

    pred_df = matn_df[
        [
            "Response_ID",
            "Annotation_ID",
            "ayah_predition",
            "prediction_claimed_source2",
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
        "ayah_predition": "Ayah",
        "prediction_claimed_source2": "claimed_source",
    })

    pred_df = pred_df.dropna(
        subset=["Label"]
    ).reset_index(drop=True)

    return matn_df, pred_df
