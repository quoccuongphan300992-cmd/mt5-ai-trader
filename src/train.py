"""Training pipeline for RandomForest baseline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL, TradingConfig
from .features import build_features, feature_columns
from .labels import add_labels
from .validation import validate_features, validate_no_future_columns


def time_split(df: pd.DataFrame, train_size: float = 0.70, val_size: float = 0.15):
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def dataset_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def validate_training_data(df: pd.DataFrame, label_col: str = "label") -> dict:
    if len(df) < 1000:
        raise ValueError(f"Refusing to train: not enough labeled rows: {len(df)}")

    counts = df[label_col].value_counts().to_dict()
    required = {LABEL_BUY, LABEL_SELL, LABEL_HOLD}
    present = set(counts.keys())
    missing = required - present

    if missing:
        raise ValueError(
            f"Refusing to train: missing labels {sorted(missing)}. "
            f"Distribution: {counts}"
        )

    if counts.get(LABEL_BUY, 0) < 50:
        raise ValueError(f"Refusing to train: too few {LABEL_BUY} labels: {counts}")

    if counts.get(LABEL_SELL, 0) < 50:
        raise ValueError(f"Refusing to train: too few {LABEL_SELL} labels: {counts}")

    return counts


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _atomic_write_model_and_metadata(bundle: dict, metadata: dict, cfg: TradingConfig) -> None:
    model_path = Path(cfg.model_path)
    metadata_path = Path(cfg.metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_model_path = model_path.with_name(f"{model_path.stem}.tmp{model_path.suffix}")
    tmp_metadata_path = metadata_path.with_name(f"{metadata_path.stem}.tmp{metadata_path.suffix}")

    joblib.dump(bundle, tmp_model_path)
    with tmp_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    os.replace(tmp_model_path, model_path)
    os.replace(tmp_metadata_path, metadata_path)


def train_model_from_dataframe(
    raw_df: pd.DataFrame,
    cfg: TradingConfig,
    feature_cols: list[str] | None = None,
    save_artifacts: bool = False,
) -> dict:
    """Train RandomForest from supplied dataframe, optionally saving model artifacts."""
    if "label" in raw_df.columns:
        labeled = raw_df.copy().reset_index(drop=True)
    else:
        featured = build_features(raw_df)
        labeled = add_labels(featured, cfg.symbol, cfg.horizon, cfg.pip_threshold)

    label_distribution = validate_training_data(labeled)
    cols = feature_cols or feature_columns(labeled)
    validate_no_future_columns(cols)
    x_train, y_train = labeled[cols], labeled["label"]
    validate_features(x_train, cols)

    model = _random_forest()
    model.fit(x_train, y_train)

    metadata = {
        "model_type": "RandomForestClassifier",
        "config": cfg.__dict__,
        "features": cols,
        "feature_count": len(cols),
        "dataset_hash": dataset_hash(raw_df),
        "rows_raw": len(raw_df),
        "rows_labeled": len(labeled),
        "label_distribution": label_distribution,
        "train_start_time": str(labeled["time"].iloc[0]),
        "train_end_time": str(labeled["time"].iloc[-1]),
        "train_rows": len(labeled),
        "classes": model.classes_.tolist(),
    }
    bundle = {"model": model, "features": cols, "config": cfg.__dict__, "metadata": metadata}
    if save_artifacts:
        _atomic_write_model_and_metadata(bundle, metadata, cfg)
    return bundle


def train_random_forest(raw_df: pd.DataFrame, cfg: TradingConfig) -> dict:
    featured = build_features(raw_df)
    labeled = add_labels(featured, cfg.symbol, cfg.horizon, cfg.pip_threshold)
    label_distribution = validate_training_data(labeled)
    cols = feature_columns(labeled)
    validate_no_future_columns(cols)

    train_df, val_df, test_df = time_split(labeled)
    x_train, y_train = train_df[cols], train_df["label"]
    x_val, y_val = val_df[cols], val_df["label"]
    x_test, y_test = test_df[cols], test_df["label"]
    validate_features(x_train, cols)
    validate_features(x_val, cols)
    validate_features(x_test, cols)

    model = _random_forest()
    model.fit(x_train, y_train)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    metadata = {
        "model_type": "RandomForestClassifier",
        "config": cfg.__dict__,
        "features": cols,
        "feature_count": len(cols),
        "dataset_hash": dataset_hash(raw_df),
        "rows_raw": len(raw_df),
        "rows_labeled": len(labeled),
        "label_distribution": label_distribution,
        "train_start_time": str(train_df["time"].iloc[0]),
        "train_end_time": str(train_df["time"].iloc[-1]),
        "validation_start_time": str(val_df["time"].iloc[0]),
        "validation_end_time": str(val_df["time"].iloc[-1]),
        "test_start_time": str(test_df["time"].iloc[0]),
        "test_end_time": str(test_df["time"].iloc[-1]),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "classes": model.classes_.tolist(),
    }

    bundle = {"model": model, "features": cols, "config": cfg.__dict__, "metadata": metadata}
    _atomic_write_model_and_metadata(bundle, metadata, cfg)

    metrics = {
        "rows": len(labeled),
        "features": len(cols),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "label_distribution": label_distribution,
        "validation_report": classification_report(y_val, val_pred, output_dict=True, zero_division=0),
        "test_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=model.classes_).tolist(),
        "classes": model.classes_.tolist(),
    }
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    importances = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    importances.sort_values("importance", ascending=False).to_csv("reports/feature_importance.csv", index=False)
    return metrics

