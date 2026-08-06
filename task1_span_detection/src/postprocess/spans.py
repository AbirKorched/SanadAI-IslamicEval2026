from src.postprocess.cleaner import clean_span



def merge_word_spans(words, text):

    merged = []

    current = None

    for w in words:

        if w["entity"] == "O":

            if current:
                merged.append(current)

            current = None
            continue


        if (
            current
            and current["entity"] == w["entity"]
        ):

            current["end"] = w["end"]

            current["score"] = min(
                current["score"],
                w["score"]
            )

        else:

            if current:
                merged.append(current)

            current = {
                "entity": w["entity"],
                "start": w["start"],
                "end": w["end"],
                "score": w["score"]
            }


    if current:
        merged.append(current)


    # ----------------------------
    # Cleaning after merge
    # ----------------------------

    final = []

    for span in merged:

        start, end = clean_span(
            text,
            span
        )

        if start < end:

            span["start"] = start
            span["end"] = end

            final.append(span)


    return final