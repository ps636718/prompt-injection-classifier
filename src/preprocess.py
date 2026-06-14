"""
preprocess.py
──────────────────────────────────────────────────────────────────────────────
Text cleaning and normalization for the Prompt Injection Classifier pipeline.

Author : Pawan Suman
Event  : DATASPRINT PS5 | NIST University Data Science Club
──────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd
import nltk

# Download NLTK resources once — silent on repeated calls
for resource in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Pre-compile regex patterns for speed ──────────────────────────────────────
_URL_PAT        = re.compile(r"http\S+|www\S+")
_NON_ALPHA_PAT  = re.compile(r"[^a-z\s]")
_MULTISPACE_PAT = re.compile(r"\s+")

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# Text columns to merge (order matters — Prompt is primary)
TEXT_COLUMNS = ["Prompt", "question1", "question2"]


# ── Public API ────────────────────────────────────────────────────────────────

def merge_text_columns(df: pd.DataFrame) -> pd.Series:
    """
    Merge multiple text columns into a single clean_text series.

    Only merges columns that actually exist in df. Treats NaN as empty string.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series — one merged string per row
    """
    cols = [c for c in TEXT_COLUMNS if c in df.columns]
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


def clean_text(text: str) -> str:
    """
    Normalize a raw text prompt for NLP feature extraction.

    Pipeline
    --------
    1. Lowercase
    2. Remove URLs (http / www)
    3. Remove non-alphabetic characters
    4. Collapse whitespace
    5. Remove English stopwords
    6. Lemmatize remaining tokens

    Parameters
    ----------
    text : str — raw prompt

    Returns
    -------
    str — cleaned text; empty string if input is null/too short
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_PAT.sub(" ", text)
    text = _NON_ALPHA_PAT.sub(" ", text)
    text = _MULTISPACE_PAT.sub(" ", text).strip()

    tokens = [w for w in text.split() if w not in _STOP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]

    return " ".join(tokens)


def apply_cleaning(df: pd.DataFrame, col: str = "clean_text") -> pd.DataFrame:
    """
    Apply clean_text() to a dataframe column and drop short/empty rows.

    Parameters
    ----------
    df  : pd.DataFrame
    col : str — name of the text column to clean

    Returns
    -------
    pd.DataFrame — cleaned, filtered, index-reset
    """
    df = df.copy()
    df[col] = df[col].apply(clean_text)
    mask = df[col].str.strip().str.len() > 10
    return df[mask].reset_index(drop=True)


def prepare_labels(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Extract and standardise the target label into a 'clean_label' column.

    Supports 'isMalicious' and 'is_duplicate' source columns.
    Unknown / missing test labels are set to -1.

    Parameters
    ----------
    df      : pd.DataFrame
    is_test : bool — if True, unknown labels are allowed (set to -1)

    Returns
    -------
    pd.DataFrame with 'clean_label' column (int)
    """
    df = df.copy()

    if "isMalicious" in df.columns:
        if not is_test:
            df = df[df["isMalicious"].notna()].copy()
        df["clean_label"] = df["isMalicious"].fillna(-1).astype(int)

    elif "is_duplicate" in df.columns:
        if not is_test:
            df = df[df["is_duplicate"].notna()].copy()
        df["clean_label"] = df["is_duplicate"].fillna(-1).astype(int)

    else:
        df["clean_label"] = -1  # No label column — test set

    return df


def impute_numeric(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str] = ("Length", "Perplexity"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Median-impute numeric columns using training set statistics.

    Parameters
    ----------
    df_train : pd.DataFrame
    df_test  : pd.DataFrame
    cols     : iterable of column names to impute

    Returns
    -------
    (df_train, df_test) — both imputed in-place copies
    """
    df_train, df_test = df_train.copy(), df_test.copy()
    for col in cols:
        if col in df_train.columns:
            median = df_train[col].median()
            df_train[col].fillna(median, inplace=True)
            if col in df_test.columns:
                df_test[col].fillna(median, inplace=True)
    return df_train, df_test
