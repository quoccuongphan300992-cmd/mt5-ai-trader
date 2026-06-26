"""Validation helpers for model feature schema and data safety."""
from __future__ import annotations

import pandas as pd


def validate_features(df: pd.DataFrame, expected_features: list[str]) -> None:
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    non_numeric = [c for c in expected_features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric model feature columns: {non_numeric}")
    bad = df[expected_features].isna().sum()
    bad = bad[bad > 0]
    if not bad.empty:
        raise ValueError(f"NaN values in model features: {bad.to_dict()}")


def validate_no_future_columns(features: list[str]) -> None:
    forbidden = {"future_close", "future_return_pips", "label"}
    leaked = [c for c in features if c in forbidden]
    if leaked:
        raise ValueError(f"Future/leakage columns found in features: {leaked}")
