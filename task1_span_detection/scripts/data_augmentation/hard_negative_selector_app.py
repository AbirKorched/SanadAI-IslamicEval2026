import os
import json
import glob

import pandas as pd
import streamlit as st
from pyarabic.araby import strip_diacritics


DATA_GLOB = "data/shamela-raw/pages*.jsonl"
OUTPUT_JSONL = "selected_hadith_pages.jsonl"


@st.cache_data
def load_data():
    files = sorted(glob.glob(DATA_GLOB))

    dfs = []
    for f in files:
        dfs.append(pd.read_json(f, lines=True))

    df = pd.concat(dfs, ignore_index=True)

    df["body"] = df["body"].fillna("").apply(strip_diacritics)

    df = df[df["body"].str.contains("قال", regex=False)]

    df = df.reset_index(drop=True)

    return df


def load_selected_ids():
    if not os.path.exists(OUTPUT_JSONL):
        return set()

    ids = set()

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ids.add(obj["page_id"])

    return ids


def save_example(row):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        json.dump(row.to_dict(), f, ensure_ascii=False)
        f.write("\n")


df = load_data()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

selected_ids = load_selected_ids()

df = df[~df["page_id"].isin(selected_ids)].reset_index(drop=True)


st.title("Hadith Candidate Picker")

st.write(f"Remaining examples: **{len(df)}**")


if len(df) == 0:
    st.success("No remaining examples.")
    st.stop()


if "idx" not in st.session_state:
    st.session_state.idx = 0


if st.session_state.idx >= len(df):
    st.success("Finished!")
    st.stop()


row = df.iloc[st.session_state.idx]

st.markdown(f"### Example {st.session_state.idx + 1}/{len(df)}")

st.write(f"**Book ID:** {row['book_id']}")
st.write(f"**Page ID:** {row['page_id']}")

st.text_area(
    "Body",
    value=row["body"],
    height=500,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Accept", use_container_width=True):
        save_example(row)
        st.session_state.idx += 1
        st.rerun()

with col2:
    if st.button("⏭ Skip", use_container_width=True):
        st.session_state.idx += 1
        st.rerun()