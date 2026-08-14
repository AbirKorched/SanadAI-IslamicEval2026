import json
from collections import defaultdict

import pandas as pd

from src.common.normalization import normalize_quran
from src.common.normalization import strip_annotation_diacritics


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

    df_ann["text"] = df_ann["text"].apply(strip_annotation_diacritics)

    return df, df_ann


def load_quran(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    surahs = defaultdict(list)
    for item in data:
        surahs[item["surah_id"]].append(item)

    quran_surahs = []
    for surah_id in sorted(surahs.keys()):
        ayahs = sorted(surahs[surah_id], key=lambda x: x["ayah_id"])
        ayah_texts = [a["ayah_text"] for a in ayahs]

        quran_surahs.append({
            "surah_id": surah_id,
            "surah_name_ar": ayahs[0]["surah_name"],
            "num_ayahs": len(ayah_texts),
            "ayahs": ayah_texts,
            "text": " ".join(ayah_texts),
        })

    df_surah = pd.DataFrame(quran_surahs)
    df_surah["text_clean"] = df_surah["text"].apply(normalize_quran)

    return data, df_surah
