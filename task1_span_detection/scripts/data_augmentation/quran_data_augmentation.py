import pandas as pd
from datasets import load_dataset
import json
import os
import random
import re


# =====================================================
# Load Quran
# =====================================================

ds_quran = load_dataset("quranlab/quran")

df_quran = pd.DataFrame(ds_quran["train"])

df_quran = df_quran.sort_values(
    ["surah", "ayah"]
)


# =====================================================
# Arabic cleaning
# =====================================================

def clean_ar(text):

    text = str(text)

    # remove tashkeel
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u0652]",
        "",
        text
    )

    text = text.replace(
        "ٱ",
        "ا"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



df_quran["text_clean"] = (
    df_quran["text"]
    .apply(clean_ar)
)



# =====================================================
# Quran starters
# =====================================================

quran_starters = [

    "قال الله تعالى:\n\n",

    "قال تعالى:\n\n",

    "قال سبحانه وتعالى:\n\n",

    "قال عز وجل:\n\n",

    "الدليل من القرآن الكريم:\n\n",

    "ومن الأدلة على ذلك قول الله تعالى:\n\n",

    "والدليل قوله تعالى:\n\n",

    "جاء في القرآن الكريم:\n\n",

    "ورد في كتاب الله تعالى:\n\n",

    "في القرآن الكريم يقول الله تعالى:\n\n",

    "ومن الآيات الدالة على ذلك:\n\n",

    "نستدل على ذلك بقوله تعالى:\n\n",

]



# =====================================================
# Ayah wrappers
# =====================================================

ayah_wrappers = [

    ("﴿", "﴾"),

    ("\"", "\""),

    ("«", "»"),

    ("<", ">"),

    ("", ""),

]


def wrap_ayah(text):

    left, right = random.choice(
        ayah_wrappers
    )

    return (
        left + text + random.choice([' ', '.', '**' , '*', '-']) + right
    )



# =====================================================
# Source templates
# =====================================================

source_templates = [

    "سورة {surah}، الآية {ayah}",

    "سورة {surah} الآية {ayah}",

    "سورة {surah}: الآية {ayah}",

    "من سورة {surah}، الآية {ayah}",

    "في سورة {surah}، الآية {ayah}",

    "(سورة {surah}، الآية {ayah})",

    "[سورة {surah}: {ayah}]",

    "المصدر: سورة {surah}، الآية {ayah}",

    "راجع سورة {surah} الآية {ayah}",

    "انظر: سورة {surah} الآية {ayah}",

    "الآية رقم {ayah} من سورة {surah}",

]



def arabic_number(n):

    return str(n).translate(
        str.maketrans(
            "0123456789",
            "٠١٢٣٤٥٦٧٨٩"
        )
    )



def format_number(n):

    return random.choice(
        [
            str(n),
            arabic_number(n)
        ]
    )



def build_source(row):

    template = random.choice(
        source_templates
    )

    return template.format(
        surah=row["surah_name_ar"],
        ayah=format_number(row["ayah"])
    )



# =====================================================
# Generate
# =====================================================

json_records = []
tsv_records = []


counter = 0


# =====================================================
# Generate per Ayah and merge randomly
# =====================================================

json_records = []
tsv_records = []

counter = 0


# group by surah
for surah, group in df_quran.groupby("surah"):

    group = group.sort_values("ayah").reset_index(drop=True)


    idx = 0

    while idx < len(group):


        # random number of ayat to merge
        merge_size = random.choices(
            [
                1,
                2,
                3,
                4,
                5
            ],
            weights=[
                60,
                25,
                10,
                3,
                2
            ]
        )[0]


        chunk = group.iloc[
            idx:min(
                idx + merge_size,
                len(group)
            )
        ]


        response_id = (
            f"QURAN_{counter:06d}"
        )


        parts = []

        current_pos = 0

        annotations = []


        starter = random.choice(
            quran_starters
        )


        parts.append(
            starter
        )

        current_pos += len(
            starter
        )


        # sometimes source before all ayat

        add_global_source = random.random() < 0.15


        if add_global_source:


            first = chunk.iloc[0]

            source = build_source(
                first
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


            current_pos += len(source)

            parts.append("\n\n")

            current_pos += 2



        # ==========================
        # Add ayat
        # ==========================

        for _, row in chunk.iterrows():


            verse = row["text_clean"]


            ayah_text = wrap_ayah(
                verse
            )


            # Ayah span inside wrapper

            ayah_start = (
                current_pos
                +
                ayah_text.find(
                    verse
                )
            )


            parts.append(
                ayah_text
            )


            annotations.append(
                {
                    "type":
                        "Ayah",

                    "start":
                        ayah_start,

                    "end":
                        ayah_start +
                        len(verse)
                }
            )


            current_pos += len(
                ayah_text
            )


            # source after ayah
            if random.random() < 0.45:


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


            parts.append(
                "\n\n"
            )

            current_pos += 2



        generated_answer = "".join(
            parts
        )



        json_records.append(
            {
                "id":
                    response_id,

                "question_id":
                    f"Q_{counter:06d}",

                "question":
                    "اذكر الدليل من القرآن الكريم",

                "generated_answer":
                    generated_answer
            }
        )



        for ann_id, ann in enumerate(
            annotations
        ):

            tsv_records.append(
                {
                    "Response_ID":
                        response_id,

                    "Annotation_ID":
                        ann_id + 1,

                    "Segment_Type":
                        ann["type"],

                    "Span_Start":
                        ann["start"],

                    "Span_End":
                        ann["end"]
                }
            )


        counter += 1


        idx += merge_size

# =====================================================
# Save
# =====================================================

os.makedirs(
    "quran_ner_data",
    exist_ok=True
)



with open(
    "data/quran_ner_data/quran_train.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for x in json_records:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False
            )
            +
            "\n"
        )



pd.DataFrame(
    tsv_records
).to_csv(
    "data/quran_ner_data/quran_train_task_1.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)



print(
    "samples:",
    len(json_records)
)

print(
    pd.DataFrame(tsv_records)
    .head(10)
)