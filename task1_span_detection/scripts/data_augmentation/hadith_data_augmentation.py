import pandas as pd
import json
import os
import random
import re


# ===============================
# Load
# ===============================

df = pd.read_csv(
    "data/sanadset.csv"
)



# ===============================
# Cleaning
# ===============================

def remove_nar_tags(text):

    # remove only <NAR> and </NAR>
    text = re.sub(
        r"</?NAR>",
        "",
        text,
        flags=re.IGNORECASE
    )
    
    text = re.sub(
        r"</?IDF>",
        "",
        text,
        flags=re.IGNORECASE
    )
    return text



def clean_ar(text):

    text = str(text)


    # remove narrator tags
    text = remove_nar_tags(
        text
    )


    # remove tashkeel
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u0652]",
        "",
        text
    )


    # normalize alif
    text = text.replace(
        "ٱ",
        "ا"
    )


    # normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



# ===============================
# Extract tagged sections
# ===============================

def extract_tag(text, tag):

    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.S
    )


    if match:

        return clean_ar(
            match.group(1)
        )


    return ""



# ===============================
# Claimed source templates
# ===============================

source_templates = [

    "رواه {book}",

    "أخرجه {book}",

    "في {book}",

    "المصدر: {book}",

    "رواه {book} رقم {num}",

    "أخرجه {book} حديث رقم {num}",

    "رواه {book} الحديث رقم {num}",

    "{book}، حديث رقم {num}",

    "ذكره {book} برقم {num}",

]



def build_source(row):

    template = random.choice(
        source_templates
    )

    return template.format(
        book=row["Book"],
        num=row["Num_hadith"]
    )



# ===============================
# Generate dataset
# ===============================

json_records = []
tsv_records = []


counter = 0



for _, row in df.iterrows():


    raw = row["Hadith"]


    sanad = extract_tag(
        raw,
        "SANAD"
    )


    matn = extract_tag(
        raw,
        "MATN"
    )


    if not sanad or not matn:
        continue



    response_id = (
        f"HADITH_{counter:06d}"
    )


    parts = []

    annotations = []

    current_pos = 0



    # ===========================
    # ISNAD
    # ===========================

    sanad_start = current_pos


    parts.append(
        sanad
    )


    annotations.append(
        {
            "type":
                "isnad",

            "start":
                sanad_start,

            "end":
                sanad_start +
                len(sanad)
        }
    )


    current_pos += len(
        sanad
    )



    parts.append(
        "\n\n"
    )

    current_pos += 2



    # ===========================
    # MATN
    # ===========================

    matn_start = current_pos


    parts.append(
        matn
    )


    annotations.append(
        {
            "type":
                "matn",

            "start":
                matn_start,

            "end":
                matn_start +
                len(matn)
        }
    )


    current_pos += len(
        matn
    )



    parts.append(
        "\n\n"
    )

    current_pos += 2



    # ===========================
    # CLAIMED SOURCE END
    # ===========================

    source = build_source(
        row
    )


    source_start = current_pos


    parts.append(
        source
    )


    annotations.append(
        {
            "type":
                "claimed_source",

            "start":
                source_start,

            "end":
                source_start +
                len(source)
        }
    )


    current_pos += len(
        source
    )



    generated_answer = "".join(
        parts
    )



    # safety check
    if "<NAR>" in generated_answer or "</NAR>" in generated_answer:
        print("NAR TAG FOUND")
        print(generated_answer[:300])
        continue



    json_records.append(
        {
            "id":
                response_id,

            "question_id":
                f"Q_{counter:06d}",

            "question":
                "",

            "generated_answer":
                generated_answer
        }
    )



    for i, ann in enumerate(
        annotations
    ):

        tsv_records.append(
            {
                "Response_ID":
                    response_id,

                "Annotation_ID":
                    i + 1,

                "Segment_Type":
                    ann["type"],

                "Span_Start":
                    ann["start"],

                "Span_End":
                    ann["end"]
            }
        )


    counter += 1



# ===============================
# Save
# ===============================

out_dir = (
    "data/hadith_ner_data"
)


os.makedirs(
    out_dir,
    exist_ok=True
)



with open(
    f"{out_dir}/hadith_train.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for item in json_records:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            +
            "\n"
        )



pd.DataFrame(
    tsv_records
).to_csv(
    f"{out_dir}/hadith_train_task_1.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)



print(
    "Created:",
    len(json_records),
    "Hadith samples"
)

print(
    pd.DataFrame(tsv_records).head(10)
)