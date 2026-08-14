import numpy as np
from rapidfuzz import fuzz
from tqdm.auto import tqdm


TOP_K = 2048


def fuzzy_score(query, passage):
    if not isinstance(query, str) or not isinstance(passage, str):
        return 0.0

    return fuzz.partial_ratio(query, passage) / 100.0


def retrieve_matn(
    query_texts,
    model,
    gpu_index,
    df_hadith,
    top_k=TOP_K,
    batch_size=64,
):
    retrieved_ids = []
    retrieved_ranks = []
    semantic_scores = []
    fuzzy_scores = []
    retrieved_texts = []
    retrieved_titles = []

    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype("float32")

    faiss_scores, faiss_ids = gpu_index.search(
        query_embeddings,
        top_k,
    )

    for i, query_text in tqdm(
        enumerate(query_texts),
        total=len(query_texts),
        desc="MATN retrieval",
    ):
        best_fuzzy = -1
        best_idx = None
        best_rank = None
        best_semantic = None
        best_text = None
        best_title = None

        for rank, (doc_id, semantic) in enumerate(
            zip(faiss_ids[i], faiss_scores[i]),
            start=1,
        ):
            passage = df_hadith.iloc[doc_id]["hadithTxt"]
            title = df_hadith.iloc[doc_id]["title"]

            fuzzy = fuzzy_score(
                query_text,
                passage,
            )

            # Exact original candidate-selection rule:
            # choose best candidate by fuzzy.
            if fuzzy > best_fuzzy:
                best_fuzzy = fuzzy
                best_idx = int(doc_id)
                best_rank = rank
                best_semantic = float(semantic)
                best_text = passage
                best_title = title

        retrieved_ids.append(best_idx)
        retrieved_ranks.append(best_rank)
        semantic_scores.append(best_semantic)
        fuzzy_scores.append(best_fuzzy)
        retrieved_texts.append(best_text)
        retrieved_titles.append(best_title)

    return {
        "matn_retrieved_id": retrieved_ids,
        "matn_retrieved_rank": retrieved_ranks,
        "matn_retrieved_title": retrieved_titles,
        "matn_retrieved_text": retrieved_texts,
        "matn_semantic_score": semantic_scores,
        "matn_fuzzy_score": fuzzy_scores,
    }


def build_gpu_index(embeddings):
    import faiss

    cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
    cpu_index.add(embeddings)

    gpu_resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(
        gpu_resources,
        0,
        cpu_index,
    )

    return gpu_index
