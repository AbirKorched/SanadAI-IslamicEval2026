TRAILING = {
    '"',
    "'",
    ".",
    ":",
    "،",
    "»",
    ")",
    "}",
    "*",
    "{",
    "[",
    "]",
    "(",
    "<",
    ">",
    "؟",
    "!",
    "؛",
    ";",
    ",",
}


def clean_span(text, span):

    start = int(span["start"])
    end = int(span["end"])

    entity = span["entity"]


    # ----------------------------
    # Remove leading whitespace
    # ----------------------------

    while (
        start < end
        and text[start].isspace()
    ):
        start += 1


    # ----------------------------
    # isnad: وعن -> عن
    # ----------------------------

    if (
        entity == "isnad"
        and text[start:start+3] == "وعن"
    ):
        start += 1


    # ----------------------------
    # Remove trailing punctuation
    # ----------------------------

    while (
        end > start
        and text[end-1] in TRAILING
    ):
        end -= 1


    # ----------------------------
    # Remove leading punctuation
    # ----------------------------

    while (
        start < end
        and text[start] in TRAILING
    ):
        start += 1


    return start, end