import re

import pandas as pd
from rapidfuzz import fuzz


def fuzzy_search(query, df_surah, threshold_percentage=70):
    if len(query.split()) == 0:
        return {"surahs": [], "portions": []}

    matched_surahs = []
    matched_portions = []
    matched_num_ayahs = []
    matched_ids = []

    for _, row in df_surah.iterrows():
        verse_text = row["text_clean"]
        surah_name = row["surah_name_ar"]
        num_ayahs = row["num_ayahs"]
        surah_id = row["surah_id"]

        alignment = fuzz.partial_ratio_alignment(query, verse_text)

        if alignment and alignment.score >= threshold_percentage:
            start_idx = alignment.dest_start
            end_idx = alignment.dest_end
            matched_substring = verse_text[start_idx:end_idx]

            matched_surahs.append(surah_name)
            matched_portions.append(matched_substring)
            matched_num_ayahs.append(num_ayahs)
            matched_ids.append(surah_id)

    return {
        "surahs": matched_surahs,
        "portions": matched_portions,
        "num_ayahs": matched_num_ayahs,
        "surah_ids": matched_ids,
    }


def analyze_match_errors(row):
    query = row["query"]
    portions = row["matched_substring"]

    if not portions or len(portions) == 0:
        return pd.Series({
            "error_type": "No Match Found",
            "is_error_accepted": False,
        })

    matched_portion = portions[0]

    if query == matched_portion:
        return pd.Series({
            "error_type": "Exact Character Match",
            "is_error_accepted": True,
        })

    q_chars = set(re.sub(r"\s+", "", query))
    v_chars = set(re.sub(r"\s+", "", matched_portion))
    differing_chars = q_chars.symmetric_difference(v_chars)

    allowed_variants = set("اأإآىيئؤءةه")
    unallowed_errors = differing_chars - allowed_variants

    if len(unallowed_errors) == 0:
        return pd.Series({
            "error_type": (
                "Permissible Script Variant "
                f"({''.join(differing_chars)})"
            ),
            "is_error_accepted": True,
        })

    query_clean = re.sub(r"\s+", "", query)
    matched_clean = re.sub(r"\s+", "", matched_portion)

    for char in unallowed_errors:
        if query_clean.endswith(char) and not matched_clean.endswith(char):
            return pd.Series({
                "error_type": (
                    "Missing Allowed Boundary Variant "
                    f"({''.join(unallowed_errors)})"
                ),
                "is_error_accepted": True,
            })

        if query_clean.startswith(char) and not matched_clean.startswith(char):
            return pd.Series({
                "error_type": (
                    "Missing Allowed Boundary Variant "
                    f"({''.join(unallowed_errors)})"
                ),
                "is_error_accepted": True,
            })

    return pd.Series({
        "error_type": f"Hard Typo / Word Change ({''.join(unallowed_errors)})",
        "is_error_accepted": False,
    })


def build_ayah_sanity_check(df_ann, df_surah):
    queries = [
        __import__("src.common.normalization", fromlist=["normalize_quran"])
        .normalize_quran(user_text)
        for user_text in df_ann[df_ann["Segment_Type"] == "Ayah"]["text"].to_list()
    ]

    # Exact adaptive threshold from the original implementation.
    thresholds = [
        (
            95
            if len(query.split()) <= 2
            else min(
                100,
                max(
                    0,
                    int(
                        (
                            (len(query) - len(query.split()))
                            / len(query)
                        ) * 100
                    ),
                ) + 10,
            )
        )
        for query in queries
    ]

    search_results = [
        fuzzy_search(
            query,
            df_surah,
            threshold_percentage=t,
        )
        for query, t in zip(queries, thresholds)
    ]

    df_sanity_check = pd.DataFrame({
        "query": queries,
        "exact_match": [
            df_surah[
                df_surah["text_clean"].str.contains(
                    query,
                    regex=False,
                )
            ]["surah_name_ar"].to_list()
            for query in queries
        ],
        "fuzzy_search": [res["surahs"] for res in search_results],
        "matched_substring": [res["portions"] for res in search_results],
        "matched_num_ayahs": [res["num_ayahs"] for res in search_results],
        "matched_surah_ids": [res["surah_ids"] for res in search_results],
        "threshold_percentage": thresholds,
    })

    df_sanity_check["prediction"] = (
        df_sanity_check["fuzzy_search"]
        .map(len)
        .apply(lambda x: "incorrect" if x == 0 else "correct")
    )

    analysis_cols = df_sanity_check.apply(
        analyze_match_errors,
        axis=1,
    )
    df_sanity_check = pd.concat(
        [df_sanity_check, analysis_cols],
        axis=1,
    )

    invalid_match_mask = (
        (df_sanity_check["prediction"] == "correct")
        & (df_sanity_check["is_error_accepted"] == False)
    )

    df_sanity_check.loc[
        invalid_match_mask,
        "prediction",
    ] = "incorrect"

    return df_sanity_check
