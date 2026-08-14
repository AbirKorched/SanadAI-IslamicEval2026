import re

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio

from src.common.normalization import normalize_quran, normalize_surah


SOURCE_LABELS = [
    "اسم السورة",
    "رقم الآية",
]


def extract_quran_source(text, ner_model):
    if not isinstance(text, str):
        return []

    normalized_text = normalize_surah(text)

    entities = ner_model.predict_entities(
        "سوره " + normalized_text
        if "سوره" not in normalized_text
        else normalized_text,
        SOURCE_LABELS,
        threshold=0.5,
    )

    return [
        {
            "text": entity["text"],
            "label": entity["label"],
        }
        for entity in entities
    ]


def extract_first(values):
    if not values:
        return None

    for v in values:
        if v is None:
            continue

        v = str(v).strip()
        if not v:
            continue

        v = re.sub(r"\s*\([^)]*\)", "", v)
        v = re.split(r"\s*[:-]\s*", v, maxsplit=1)[0]
        v = v.strip()

        if v:
            return v

    return None


def clean_surah_name(name):
    if name is None:
        return None

    name = str(name).strip()
    name = re.sub(
        r"^(?:سورة|سوره|السورة)\s*",
        "",
        name,
    )
    return name.strip()


def clean_ayah(name):
    if name is None:
        return None

    name = str(name).strip()
    name = re.sub(
        r"^(?:ايه|الايه)\s*",
        "",
        name,
    )
    return name.strip()


def quran_span_similarity(query_text, verse_text):
    query_tokens = set(query_text.split())
    verse_tokens = set(verse_text.split())

    if not query_tokens:
        return 0

    intersection = query_tokens.intersection(verse_tokens)
    token_overlap = len(intersection) / len(query_tokens)
    edit_score = ratio(query_text, verse_text) / 100

    return 0.7 * token_overlap + 0.3 * edit_score


def evaluate_claimed_source(row, surahs):
    entities = row["source_entities"]
    gt_names = row.get("surahs_name", [])

    if pd.isna(row["Label_claimed_source"]):
        return None

    if gt_names == []:
        return "incorrect"

    if entities == []:
        for matched_surah in gt_names:
            if matched_surah in row["text_claimed_source"]:
                return "correct"
        return "incorrect"

    surah_mentions = []
    ayah_mentions = []

    for e in entities:
        if not isinstance(e, dict):
            continue

        label = e.get("label", "")
        text = str(e.get("text", "")).strip()

        if label == "اسم السورة":
            surah_mentions.append(text)
        elif label == "رقم الآية":
            ayah_mentions.append(text)

    gt_ids = row.get("surah_ids", [])
    matched_num_ayahs = row.get("matched_num_ayahs", [])

    if surah_mentions:
        found = False

        for mention in surah_mentions:
            mention = (
                mention
                .replace("السوره", "")
                .replace("سوره", "")
                .strip()
            )

            if mention != "" and mention in gt_names:
                found = True
                break

            try:
                if int(mention) in gt_ids:
                    found = True
                    break
            except Exception:
                pass

        if not found:
            return "incorrect"

    if ayah_mentions:
        try:
            ayah_num = int(
                extract_first([
                    clean_ayah(ayah)
                    for ayah in ayah_mentions
                ])
            )
        except Exception:
            return "incorrect"

        valid = False

        for n in matched_num_ayahs:
            try:
                if ayah_num <= int(n):
                    valid = True
                    break
            except Exception:
                continue

        if not valid:
            return "incorrect"

    if surah_mentions and ayah_mentions:
        surah = clean_surah_name(
            extract_first(surah_mentions)
        )

        try:
            ayah_num = int(
                extract_first([
                    clean_ayah(ayah)
                    for ayah in ayah_mentions
                ])
            )
        except Exception:
            return "incorrect"

        verse = None

        try:
            sid = int(surah)

            for s in surahs:
                if (
                    s["surah_id"] == sid
                    and s["ayah_id"] == ayah_num
                ):
                    verse = s
                    break

        except Exception:
            for s in surahs:
                if (
                    normalize_surah(s["surah_name"]) == surah
                    and s["ayah_id"] == ayah_num
                ):
                    verse = s
                    break

        if verse is None:
            return "incorrect"

        verse_text = normalize_quran(verse["ayah_text"])
        query_text = normalize_quran(str(row["text"]))

        verse_text = " ".join(verse_text.split())
        query_text = " ".join(query_text.split())

        score = quran_span_similarity(
            query_text,
            verse_text,
        )

        if query_text in verse_text:
            return "correct"

        if score < 0.3:
            return "incorrect"

    return "correct"


def binary_source_score(row):
    text = row["text_claimed_source"]
    surahs = row["surahs_name"]

    if pd.isna(text) or not isinstance(surahs, list):
        return np.nan

    text_norm = re.sub(r"\bال(?=\S)", "", str(text))
    text_norm = re.sub(r"\s+", " ", text_norm).strip()

    for surah in surahs:
        surah_norm = re.sub(r"\bال(?=\S)", "", str(surah))
        surah_norm = re.sub(r"\s+", " ", surah_norm).strip()

        if surah_norm in text_norm:
            return "correct"

    return "incorrect"
