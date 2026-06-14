"""
features.py
──────────────────────────────────────────────────────────────────────────────
Feature engineering pipeline: TF-IDF + domain-specific meta-features.

Final feature matrix shape: (n_samples, 15_019)
  ├── 15,000  TF-IDF character n-gram features
  ├──      1  special character density
  ├──      2  word_count + char_count
  └──     18  jailbreak keyword flags (binary)

Author : Pawan Suman
Event  : DATASPRINT PS5 | NIST University Data Science Club
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# ── Jailbreak Keyword Patterns ────────────────────────────────────────────────
# 18 hand-curated adversarial signals — domain knowledge the TF-IDF
# would only learn implicitly from rare training examples.

JAILBREAK_KEYWORDS: list[str] = [
    "jailbreak",
    "dan ",
    "do anything now",
    "ignore previous",
    "ignore all",
    "bypass",
    "system prompt",
    "disregard",
    "forget your instructions",
    "you are now",
    "pretend you",
    "act as if",
    "roleplay as",
    "you have no restrictions",
    "override",
    "developer mode",
    "sudo",
    "unrestricted",
]


# ── TF-IDF Vectorizer ─────────────────────────────────────────────────────────

def build_tfidf(
    train_texts: pd.Series,
    test_texts: pd.Series,
    ngram_range: tuple = (1, 2),
    max_features: int = 15_000,
) -> tuple:
    """
    Fit TF-IDF on training text and transform both splits.

    Parameters
    ----------
    train_texts  : pd.Series — cleaned training text
    test_texts   : pd.Series — cleaned test text
    ngram_range  : tuple — (min_n, max_n) for n-gram extraction
    max_features : int   — vocabulary size cap

    Returns
    -------
    (tfidf, X_train_text, X_test_text)
      tfidf         : fitted TfidfVectorizer
      X_train_text  : sparse matrix (n_train, max_features)
      X_test_text   : sparse matrix (n_test,  max_features)
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,   # log(1+tf) — dampens keyword repetition attacks
        min_df=2,            # ignore hapax legomena
    )
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)
    return tfidf, X_train, X_test


# ── Keyword Flags ─────────────────────────────────────────────────────────────

def keyword_flags(texts: pd.Series) -> np.ndarray:
    """
    Build a binary (0/1) flag matrix for each jailbreak keyword.

    Shape: (len(texts), len(JAILBREAK_KEYWORDS))
    """
    lower_texts = texts.str.lower().fillna("")
    rows = [
        [1 if kw in text else 0 for kw in JAILBREAK_KEYWORDS]
        for text in lower_texts
    ]
    return np.array(rows, dtype=np.float32)


# ── Special Character Density ─────────────────────────────────────────────────

def special_char_density(texts: pd.Series) -> np.ndarray:
    """
    Compute the ratio of non-alphanumeric (non-space) characters per prompt.

    Obfuscated jailbreak attempts often contain higher symbol density.
    Shape: (len(texts), 1)
    """
    def _density(text: str) -> float:
        if not text or len(text) == 0:
            return 0.0
        n_special = sum(1 for c in text if not c.isalnum() and c != " ")
        return n_special / len(text)

    values = texts.fillna("").apply(_density).values.astype(np.float32)
    return values.reshape(-1, 1)


# ── Structural Meta-Features ──────────────────────────────────────────────────

def structural_features(df: pd.DataFrame, text_col: str = "clean_text") -> np.ndarray:
    """
    Compute word_count and char_count with outlier clipping.

    Parameters
    ----------
    df       : pd.DataFrame
    text_col : str — column containing cleaned text

    Returns
    -------
    np.ndarray, shape (n_rows, 2) — [word_count, char_count]
    """
    word_count = df[text_col].str.split().str.len().clip(upper=100).fillna(0)
    char_count = df[text_col].str.len().clip(upper=500).fillna(0)
    return np.column_stack([word_count.values, char_count.values]).astype(np.float32)


# ── Full Meta-Feature Stack ───────────────────────────────────────────────────

def build_meta_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = "clean_text",
    scale: bool = True,
) -> tuple:
    """
    Build and optionally scale all meta-features for both splits.

    Meta-feature order:
        [word_count, char_count, special_char_density, kw_flag_0 … kw_flag_17]

    Parameters
    ----------
    df_train : pd.DataFrame
    df_test  : pd.DataFrame
    text_col : str  — name of cleaned text column
    scale    : bool — apply StandardScaler (with_mean=False for sparse compat)

    Returns
    -------
    (scaler, X_meta_train, X_meta_test)
      scaler        : fitted StandardScaler (or None if scale=False)
      X_meta_train  : sparse csr_matrix (n_train, 21)
      X_meta_test   : sparse csr_matrix (n_test,  21)
    """
    # Structural features
    struct_train = structural_features(df_train, text_col)
    struct_test  = structural_features(df_test,  text_col)

    # Special char density
    sc_train = special_char_density(df_train[text_col])
    sc_test  = special_char_density(df_test[text_col])

    # Keyword flags
    kw_train = keyword_flags(df_train[text_col])
    kw_test  = keyword_flags(df_test[text_col])

    # Stack all meta-features
    meta_train = np.hstack([struct_train, sc_train, kw_train])
    meta_test  = np.hstack([struct_test,  sc_test,  kw_test])

    scaler = None
    if scale:
        scaler = StandardScaler(with_mean=False)
        meta_train = scaler.fit_transform(meta_train)
        meta_test  = scaler.transform(meta_test)

    return scaler, sp.csr_matrix(meta_train), sp.csr_matrix(meta_test)


# ── Final Feature Matrix ──────────────────────────────────────────────────────

def build_feature_matrix(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_col: str = "clean_text",
) -> tuple:
    """
    End-to-end feature construction: TF-IDF + scaled meta-features.

    Returns
    -------
    (tfidf, scaler, X_train_full, X_test_full)
    """
    tfidf, X_text_train, X_text_test = build_tfidf(
        df_train[text_col], df_test[text_col]
    )
    scaler, X_meta_train, X_meta_test = build_meta_features(
        df_train, df_test, text_col
    )

    X_train_full = hstack([X_text_train, X_meta_train])
    X_test_full  = hstack([X_text_test,  X_meta_test])

    return tfidf, scaler, X_train_full, X_test_full
