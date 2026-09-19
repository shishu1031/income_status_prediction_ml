from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml

TARGET_RAW = "class"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"


def load_adult_dataset() -> pd.DataFrame:
    """Download the Adult dataset from OpenML and return a cleaned DataFrame."""
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_RAW not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_RAW}' was not found.")

    df[TARGET_RAW] = df[TARGET_RAW].astype(str).str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": pd.NA})

    df["income_status"] = (df[TARGET_RAW] == POSITIVE_LABEL).astype(int)
    df = df.drop(columns=[TARGET_RAW])
    return df


def split_features_target(df: pd.DataFrame):
    """Separate predictors, binary target, and an analysis copy containing target labels."""
    if "income_status" not in df.columns:
        raise ValueError("Expected 'income_status' target column.")
    X = df.drop(columns=["income_status"])
    y = df["income_status"].astype(int)
    return X, y
