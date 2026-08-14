import pandas as pd

from src.common.normalization import normalize_quran, strip_annotation_diacritics


def load_hadith(path):
    df_hadith = pd.read_json(path)
    df_hadith["text_clean"] = df_hadith["hadithTxt"].apply(
        normalize_quran
    )
    return df_hadith


def load_annotations(dev_jsonl, task_tsv):
    df = pd.read_json(dev_jsonl, lines=True)
    df_ann = pd.read_csv(task_tsv, sep="\t")

    df_ann = df_ann.merge(
        df,
        left_on="Response_ID",
        right_on="id",
        how="left",
    )

    df_ann["text"] = df_ann.apply(
        lambda row: row["generated_answer"][
            int(row["Span_Start"]):int(row["Span_End"])
        ]
        if row["Span_Start"] != "-"
        else None,
        axis=1,
    )


    df_ann["text"] = df_ann["text"].apply(
        strip_annotation_diacritics
    )

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

    isnad = (
        df_ann[df_ann["Segment_Type"] == "isnad"][
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
                "Segment_Type": "Segment_Type_isnad",
                "Label": "Label_isnad",
                "Span_Start": "Span_Start_isnad",
                "Span_End": "Span_End_isnad",
                "text": "text_isnad",
            }
        )
    )

    main = main.merge(claimed, on=keys, how="left")
    main = main.merge(isnad, on=keys, how="left")

    # Exact assertions present in the original code.
    assert main["text_isnad"].nunique() == 26
    assert main["text_claimed_source"].nunique() == 298

    return df, df_ann, main


def build_matn_df(main):
    return (
        main[main["Segment_Type"] == "matn"][
            [
                "Label",
                "text",
                "Response_ID",
                "Annotation_ID",
                "text_claimed_source",
                "Label_claimed_source",
                "text_isnad",
                "Label_isnad",
            ]
        ]
        .reset_index(drop=True)
    )
