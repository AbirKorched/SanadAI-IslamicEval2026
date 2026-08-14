import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ayah.retrieval import analyze_match_errors


def test_exact_quran_match_is_accepted():
    row = pd.Series({
        "query": "نصي",
        "matched_substring": ["نص"],
    })
    result = analyze_match_errors(row)
    print(result)
    assert result["is_error_accepted"] is True

def test_exact_quran_match_is_not_accepted():
    row = pd.Series({
        "query": "نكص",
        "matched_substring": ["نص"],
    })
    result = analyze_match_errors(row)
    print(result)
    assert result["is_error_accepted"] is False