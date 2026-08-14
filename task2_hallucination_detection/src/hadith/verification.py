import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm.auto import tqdm


SOURCE_LABELS = [
    "اسم كتاب حديث",
    "اسم محدث",
    "اسم راو",
    "مصدر الحديث",
    "اسم كتاب",
    "عدد الآية",
    "عدد السورة",
]


def normalize_entity(text):
    text = str(text).strip()

    remove_words = [
        "رواه",
        "أخرجه",
        "حدثنا",
        "الإمام",
        "في",
        "صحيح",
        "سنن",
        "كتاب",
    ]

    for w in remove_words:
        text = text.replace(w, "")

    return text.strip()


def extract_source_entities(text, ner_model):
    if not isinstance(text, str) or text.strip() == "":
        return []

    entities = ner_model.predict_entities(
        text,
        SOURCE_LABELS,
        threshold=0.5,
    )

    extracted = []

    for e in entities:
        ent = normalize_entity(e["text"])

        if len(ent) > 1:
            extracted.append(ent)

    return list(set(extracted))


def entity_max_similarity(claimed_entities, true_entities):
    if len(claimed_entities) == 0:
        return np.nan

    if len(true_entities) == 0:
        return 0.0

    scores = []

    for c in claimed_entities:
        best = 0.0

        for t in true_entities:
            sim = fuzz.token_set_ratio(c, t) / 100.0
            best = max(best, sim)

        scores.append(best)

    return float(np.mean(scores))


def compute_claimed_source_scores(
    matn_df,
    df_hadith,
    model,
    ner_model,
):
    source_entity_scores = []
    source_semantic_scores = []

    for _, row in tqdm(
        matn_df.iterrows(),
        total=len(matn_df),
        desc="Source verification",
    ):
        claimed_source = row["text_claimed_source"]
        label_source = row["Label_claimed_source"]
        retrieved_id = row["matn_retrieved_id"]

        if pd.isna(label_source):
            source_entity_scores.append(np.nan)
            source_semantic_scores.append(np.nan)
            continue

        if (
            pd.isna(claimed_source)
            or claimed_source == ""
            or pd.isna(retrieved_id)
        ):
            source_entity_scores.append(np.nan)
            source_semantic_scores.append(np.nan)
            continue

        retrieved_row = df_hadith.iloc[int(retrieved_id)]
        true_source = str(retrieved_row["title"])

        claimed_entities = extract_source_entities(
            claimed_source,
            ner_model,
        )
        true_entities = extract_source_entities(
            true_source,
            ner_model,
        )

        entity_score = entity_max_similarity(
            claimed_entities,
            true_entities,
        )

        emb = model.encode(
            [claimed_source, true_source],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        semantic = float(np.dot(emb[0], emb[1]))

        source_entity_scores.append(entity_score)
        source_semantic_scores.append(semantic)

    matn_df["source_entity_score"] = source_entity_scores
    matn_df["source_semantic_score"] = source_semantic_scores

    return matn_df


def compute_isnad_scores(
    matn_df,
    df_hadith,
    model,
    ner_model,
):
    isnad_entity_scores = []
    isnad_semantic_scores = []

    for _, row in tqdm(
        matn_df.iterrows(),
        total=len(matn_df),
        desc="Source verification",
    ):
        isnad = row["text_isnad"]
        label_isnad = row["Label_isnad"]
        retrieved_id = row["matn_retrieved_id"]

        if pd.isna(label_isnad):
            isnad_entity_scores.append(np.nan)
            isnad_semantic_scores.append(np.nan)
            continue

        if (
            pd.isna(isnad)
            or isnad == ""
            or pd.isna(retrieved_id)
        ):
            isnad_entity_scores.append(np.nan)
            isnad_semantic_scores.append(np.nan)
            continue

        retrieved_row = df_hadith.iloc[int(retrieved_id)]
        true_isnad = str(retrieved_row["hadithTxt"])

        isnad_entities = extract_source_entities(
            isnad,
            ner_model,
        )
        true_entities = extract_source_entities(
            true_isnad,
            ner_model,
        )

        entity_score = entity_max_similarity(
            isnad_entities,
            true_entities,
        )

        emb = model.encode(
            [isnad, true_isnad],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        semantic = float(np.dot(emb[0], emb[1]))

        isnad_entity_scores.append(entity_score)
        isnad_semantic_scores.append(semantic)

    matn_df["isnad_entity_score"] = isnad_entity_scores
    matn_df["isnad_semantic_score"] = isnad_semantic_scores

    return matn_df
