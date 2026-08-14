import re

from pyarabic.araby import strip_diacritics, strip_tashkeel


def normalize_surah(text):
    text = strip_tashkeel(text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_quran(text):
    text = strip_tashkeel(text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ـ", "")
    text = re.sub(r"[\u06D6-\u06ED]", "", text)
    text = re.sub(r"[^\u0621-\u063A\u0641-\u064A\u0670 ]+", "", text)
    text = re.sub(r"\u0670", "ا", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_arabic(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\bال(?=\S)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_annotation_diacritics(text):
    return strip_diacritics(text)
