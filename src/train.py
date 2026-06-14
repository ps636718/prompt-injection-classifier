"""
train.py
──────────────────────────────────────────────────────────────────────────────
Model training, hyperparameter tuning, ensemble construction, and evaluation.

Final Result:
  Model     : Ensemble VotingClassifier (SVC + LR + RF)
  Accuracy  : 0.9482
  F1 Score  : 0.9464

Author : Pawan Suman
Event  : DATASPRINT PS5 | NIST University Data Science Club
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
)


# ── Train / Validation Split ──────────────────────────────────────────────────

def make_split(
    X, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """
    Stratified 80/20 train-validation split.

    Parameters
    ----------
    X            : feature matrix (sparse or dense)
    y            : label array
    test_size    : float — fraction for validation
    random_state : int

    Returns
    -------
    (X_train, X_val, y_train, y_val)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ── Individual Models ─────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """
    Return a dict of all base classifiers with balanced class weights.

    All models use class_weight='balanced' to handle label imbalance —
    a common issue in security datasets where benign >> malicious.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
        ),
        "LinearSVC": LinearSVC(
            random_state=42,
            max_iter=3000,
            C=1.5,
            class_weight="balanced",
        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def evaluate_models(models: dict, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """
    Train all models and return a results DataFrame sorted by F1 Score.

    Parameters
    ----------
    models  : dict {name: classifier}
    X_train : training features
    y_train : training labels
    X_val   : validation features
    y_val   : validation labels

    Returns
    -------
    pd.DataFrame with columns [Model, Accuracy, Precision, F1 Score]
    """
    rows = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_val, pred), 4),
            "Precision": round(precision_score(y_val, pred, average="binary", zero_division=0), 4),
            "F1 Score":  round(f1_score(y_val, pred, average="binary"), 4),
        })

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    return results


# ── Hyperparameter Tuning ─────────────────────────────────────────────────────

def tune_logistic_regression(X, y) -> tuple:
    """
    GridSearchCV over C for Logistic Regression (cv=3, scoring='f1').

    Returns (best_estimator, best_C, best_cv_f1)
    """
    gs = GridSearchCV(
        LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
        ),
        param_grid={"C": [0.1, 0.5, 1.0, 1.5, 5.0, 10.0]},
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_["C"], gs.best_score_


def tune_linear_svc(X_train, y_train) -> tuple:
    """
    GridSearchCV over C and loss for LinearSVC (cv=3, scoring='f1').

    Returns (best_estimator, best_params, best_cv_f1)
    """
    gs = GridSearchCV(
        LinearSVC(
            random_state=42,
            max_iter=2000,
            class_weight="balanced",
        ),
        param_grid={
            "C":    [0.1, 1, 5, 10, 50],
            "loss": ["hinge", "squared_hinge"],
        },
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_ensemble(
    svc_tuned,
    lr_balanced: LogisticRegression,
    rf_balanced: RandomForestClassifier,
) -> VotingClassifier:
    """
    Build a hard-voting ensemble from three trained classifiers.

    Hard voting is used because LinearSVC does not natively support
    predict_proba (required for soft voting).

    Parameters
    ----------
    svc_tuned    : fitted LinearSVC (tuned)
    lr_balanced  : fitted LogisticRegression
    rf_balanced  : fitted RandomForestClassifier

    Returns
    -------
    Fitted VotingClassifier
    """
    return VotingClassifier(
        estimators=[
            ("svc", svc_tuned),
            ("lr",  lr_balanced),
            ("rf",  rf_balanced),
        ],
        voting="hard",
    )


# ── Final Evaluation ──────────────────────────────────────────────────────────

def full_report(model, X_val, y_val, model_name: str = "Ensemble") -> None:
    """
    Print classification report and key metrics for a fitted model.
    """
    y_pred = model.predict(X_val)

    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_val, y_pred,
        target_names=["Benign (0)", "Malicious (1)"],
    ))
    print(f"  Accuracy  : {accuracy_score(y_val, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_val, y_pred, average='binary'):.4f}")
    print(f"{'='*60}\n")
