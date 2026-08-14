import pandas as pd


def merge_predictions(quran_path, hadith_path, output_path):
    df_pred_quran = pd.read_csv(quran_path, sep="\t")
    df_pred_hadith = pd.read_csv(hadith_path, sep="\t")

    df_pred = pd.concat(
        [df_pred_quran, df_pred_hadith],
        ignore_index=True,
    )

    df_pred = df_pred.sort_values(
        by=["Response_ID", "Annotation_ID"],
        ascending=[True, True],
    ).reset_index(drop=True)

    df_pred.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    return df_pred
