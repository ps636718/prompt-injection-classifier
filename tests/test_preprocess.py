"""
tests/test_preprocess.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the text preprocessing pipeline.

Run with:
    pytest tests/ -v

Author : Pawan Suman
Event  : DATASPRINT PS5 | NIST University Data Science Club
──────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import (
    clean_text,
    merge_text_columns,
    apply_cleaning,
    prepare_labels,
    impute_numeric,
)


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_url_removed(self):
        result = clean_text("visit http://evil.com now")
        assert "http" not in result
        assert "evil" not in result  # URL domain stripped

    def test_non_alpha_stripped(self):
        result = clean_text("hello!!! world@@@")
        assert "!" not in result
        assert "@" not in result

    def test_stopwords_removed(self):
        result = clean_text("the quick brown fox")
        # 'the' is a stopword and should be removed
        assert "the" not in result.split()

    def test_lemmatization(self):
        result = clean_text("running foxes")
        # lemmatizer should normalise "running" → "running" or "run"
        # and "foxes" → "fox"
        assert "fox" in result

    def test_null_input(self):
        assert clean_text(None) == ""
        assert clean_text(float("nan")) == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_short_but_valid(self):
        # Short text with a real word — should return something
        result = clean_text("hack me")
        assert isinstance(result, str)


# ── merge_text_columns ────────────────────────────────────────────────────────

class TestMergeTextColumns:
    def test_merges_present_columns(self):
        df = pd.DataFrame({
            "Prompt":    ["hello"],
            "question1": ["world"],
            "question2": ["foo"],
        })
        result = merge_text_columns(df)
        assert result[0] == "hello world foo"

    def test_handles_missing_column(self):
        df = pd.DataFrame({"Prompt": ["only this"]})
        result = merge_text_columns(df)
        assert result[0] == "only this"

    def test_nan_treated_as_empty(self):
        df = pd.DataFrame({
            "Prompt":    ["hi"],
            "question1": [None],
        })
        result = merge_text_columns(df)
        assert "None" not in result[0]


# ── apply_cleaning ────────────────────────────────────────────────────────────

class TestApplyCleaning:
    def test_drops_short_rows(self):
        df = pd.DataFrame({"clean_text": ["hi", "this is a longer valid prompt text"]})
        result = apply_cleaning(df)
        assert len(result) == 1  # "hi" should be dropped (len <= 10)

    def test_resets_index(self):
        df = pd.DataFrame({"clean_text": ["x", "this is a properly long enough string"]})
        result = apply_cleaning(df)
        assert list(result.index) == list(range(len(result)))


# ── prepare_labels ────────────────────────────────────────────────────────────

class TestPrepareLabels:
    def test_isMalicious_column(self):
        df = pd.DataFrame({"isMalicious": [0, 1, 0, 1]})
        result = prepare_labels(df)
        assert "clean_label" in result.columns
        assert list(result["clean_label"]) == [0, 1, 0, 1]

    def test_is_duplicate_fallback(self):
        df = pd.DataFrame({"is_duplicate": [1, 0]})
        result = prepare_labels(df)
        assert list(result["clean_label"]) == [1, 0]

    def test_no_label_column(self):
        df = pd.DataFrame({"Prompt": ["hello", "world"]})
        result = prepare_labels(df, is_test=True)
        assert all(result["clean_label"] == -1)

    def test_drops_nan_labels_in_train(self):
        df = pd.DataFrame({"isMalicious": [0.0, None, 1.0]})
        result = prepare_labels(df, is_test=False)
        assert len(result) == 2  # NaN row dropped


# ── impute_numeric ────────────────────────────────────────────────────────────

class TestImputeNumeric:
    def test_median_imputation(self):
        train = pd.DataFrame({"Length": [10.0, 20.0, None]})
        test  = pd.DataFrame({"Length": [None, 5.0]})
        tr_out, te_out = impute_numeric(train, test, cols=["Length"])
        # Training median of [10, 20] = 15.0
        assert tr_out["Length"].isna().sum() == 0
        assert te_out["Length"].iloc[0] == 15.0

    def test_missing_column_ignored(self):
        train = pd.DataFrame({"Length": [1.0]})
        test  = pd.DataFrame({"Length": [2.0]})
        # "Perplexity" doesn't exist — should not crash
        tr_out, te_out = impute_numeric(train, test, cols=["Length", "Perplexity"])
        assert "Perplexity" not in tr_out.columns
