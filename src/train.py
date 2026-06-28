"""Training pipeline for RandomForest baseline."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # optional dependency
    XGBClassifier = None

from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL, TradingConfig
from .data import latest_raw_csv, load_csv
from .features import build_features, feature_columns
from .labels import add_labels
from .validation import validate_features, validate_no_future_columns



def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def feature_schema_hash(feature_cols: list[str]) -> str:
    payload = json.dumps(list(feature_cols), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_model_bundle(model_path: str | Path) -> dict:
    """Load and normalize a candidate model bundle from joblib."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate model missing: {path}")
    raw_bundle = joblib.load(path)
    if isinstance(raw_bundle, dict) and "model" in raw_bundle:
        model = raw_bundle["model"]
        features = raw_bundle.get("features") or raw_bundle.get("feature_columns")
        metadata = raw_bundle.get("metadata") or {}
        if features is None and isinstance(metadata, dict):
            features = metadata.get("features") or metadata.get("feature_columns")
    else:
        model = raw_bundle
        features = None
        metadata = {}
    return {
        "model": model,
        "feature_columns": list(features) if features is not None else None,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "raw_bundle": raw_bundle,
    }


def _sample_training_data(rows: int = 5000) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    time = pd.date_range("2020-01-01", periods=rows, freq="h")
    returns = rng.normal(0, 0.0007, rows)
    close = 1.10 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.integers(8, 20, rows)
    high = np.maximum(open_, close) + rng.uniform(0.00005, 0.0008, rows)
    low = np.minimum(open_, close) - rng.uniform(0.00005, 0.0008, rows)
    return pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close, "tick_volume": rng.integers(100, 2000, rows), "spread": spread, "real_volume": 0})


def _resolve_training_dataframe(csv_path: str | Path | None, sample: bool) -> pd.DataFrame:
    if sample:
        return _sample_training_data()
    chosen = str(csv_path) if csv_path else latest_raw_csv()
    if not chosen:
        raise ValueError("No CSV found. Use --csv or --sample.")
    return load_csv(chosen)


def _metadata_config(metadata: dict, symbol: str | None, timeframe: str | None, bars: int | None) -> TradingConfig:
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    winning = metadata.get("winning_config") if isinstance(metadata.get("winning_config"), dict) else {}
    return TradingConfig(
        symbol=symbol or config.get("symbol", "EURUSD"),
        timeframe=timeframe or config.get("timeframe", "H1"),
        bars=bars or config.get("bars", 100000),
        horizon=int(winning.get("horizon", config.get("horizon", 10))),
        pip_threshold=float(config.get("pip_threshold", 30.0)),
        signal_threshold=float(winning.get("threshold", metadata.get("threshold", config.get("signal_threshold", 0.75)))),
        risk_per_trade=float(config.get("risk_per_trade", 0.01)),
        trade_mode=str(config.get("trade_mode", "paper")),
        label_method=str(winning.get("label_method", config.get("label_method", "fixed_return"))),
        label_atr_tp_mult=float(winning.get("label_atr_tp_mult", config.get("label_atr_tp_mult", 1.5))),
        label_atr_sl_mult=float(winning.get("label_atr_sl_mult", config.get("label_atr_sl_mult", 1.0))),
        model_type=str(config.get("model_type", "extra_trees")),
    )


def continue_train_sklearn_ensemble(
    *,
    csv_path: str | Path | None,
    sample: bool,
    candidate_model_path: str | Path,
    candidate_metadata_path: str | Path | None,
    output_dir: str | Path,
    candidate_id: str,
    add_estimators: int = 300,
    allow_retrain_fallback: bool = False,
    symbol: str | None = None,
    timeframe: str | None = None,
    bars: int | None = None,
) -> dict:
    """Continue training an existing sklearn ensemble candidate in a new folder."""
    if add_estimators <= 0:
        raise ValueError("add_estimators must be > 0")
    parent_model_path = Path(candidate_model_path)
    parent_metadata_path = Path(candidate_metadata_path) if candidate_metadata_path else None
    bundle = load_model_bundle(parent_model_path)
    model = bundle["model"]
    metadata = dict(bundle.get("metadata") or {})
    if parent_metadata_path and parent_metadata_path.exists():
        file_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
        if isinstance(file_metadata, dict):
            metadata = {**metadata, **file_metadata}
    cfg = _metadata_config(metadata, symbol, timeframe, bars)
    raw_df = _resolve_training_dataframe(csv_path, sample)
    featured = build_features(raw_df)
    labeled = add_labels(featured, cfg.symbol, cfg.horizon, cfg.pip_threshold, cfg.label_method, cfg.label_atr_tp_mult, cfg.label_atr_sl_mult)
    label_distribution = validate_training_data(labeled)
    new_feature_columns = feature_columns(labeled)
    old_feature_columns = bundle.get("feature_columns") or metadata.get("features") or metadata.get("feature_columns")
    if old_feature_columns is not None:
        old_feature_columns = list(old_feature_columns)
    train_mode = "warm_start_continue"
    fallback_reason = None
    if old_feature_columns is not None and old_feature_columns != new_feature_columns:
        if not allow_retrain_fallback:
            raise ValueError("Feature schema mismatch. Refusing warm-start continuation. Use --allow-retrain-fallback to retrain from candidate config.")
        model = create_model(cfg.model_type)
        train_mode = "fallback_retrain"
        fallback_reason = "feature_schema_mismatch"
    elif not hasattr(model, "warm_start"):
        raise ValueError("Model does not support warm_start")
    elif not hasattr(model, "n_estimators"):
        raise ValueError("Model does not expose n_estimators")
    old_n_estimators = int(getattr(model, "n_estimators", 0))
    new_n_estimators = old_n_estimators + int(add_estimators) if train_mode == "warm_start_continue" else old_n_estimators
    if train_mode == "warm_start_continue":
        model.set_params(warm_start=True, n_estimators=new_n_estimators)
    x_train, y_train = labeled[new_feature_columns], labeled["label"]
    validate_features(x_train, new_feature_columns)
    _fit_model(model, x_train, y_train)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    new_candidate_id = f"{candidate_id}_continued_{timestamp}"
    candidate_dir = Path(output_dir) / new_candidate_id
    if candidate_dir.exists():
        raise FileExistsError(f"Output candidate directory already exists: {candidate_dir}")
    candidate_dir.mkdir(parents=True, exist_ok=False)
    model_output_path = candidate_dir / "model.joblib"
    metadata_output_path = candidate_dir / "metadata.json"
    manifest_output_path = candidate_dir / "continue_train_manifest.json"
    new_metadata = {
        **metadata,
        "candidate_id": new_candidate_id,
        "parent_candidate_id": candidate_id,
        "train_mode": train_mode,
        "fallback_reason": fallback_reason,
        "parent_model_path": str(parent_model_path),
        "parent_metadata_path": str(parent_metadata_path) if parent_metadata_path else None,
        "old_n_estimators": old_n_estimators,
        "add_estimators": int(add_estimators),
        "new_n_estimators": int(getattr(model, "n_estimators", new_n_estimators)),
        "warm_start": bool(getattr(model, "warm_start", False)),
        "features": new_feature_columns,
        "feature_columns": new_feature_columns,
        "feature_schema_hash": feature_schema_hash(new_feature_columns),
        "label_method": cfg.label_method,
        "tp": cfg.label_atr_tp_mult,
        "sl": cfg.label_atr_sl_mult,
        "horizon": cfg.horizon,
        "threshold": cfg.signal_threshold,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.__dict__,
        "dataset_hash": dataset_hash(raw_df),
        "rows_raw": len(raw_df),
        "rows_labeled": len(labeled),
        "label_distribution": label_distribution,
        "train_start_time": str(labeled["time"].iloc[0]),
        "train_end_time": str(labeled["time"].iloc[-1]),
        "classes": model.classes_.tolist(),
    }
    new_bundle = {"model": model, "features": new_feature_columns, "config": cfg.__dict__, "metadata": new_metadata}
    joblib.dump(new_bundle, model_output_path)
    metadata_output_path.write_text(json.dumps(_json_safe_value(new_metadata), indent=2), encoding="utf-8")
    manifest = {
        "candidate_id": new_candidate_id,
        "parent_candidate_id": candidate_id,
        "train_mode": train_mode,
        "fallback_reason": fallback_reason,
        "candidate_dir": str(candidate_dir),
        "model_path": str(model_output_path),
        "metadata_path": str(metadata_output_path),
        "manifest_path": str(manifest_output_path),
        "parent_model_path": str(parent_model_path),
        "old_n_estimators": old_n_estimators,
        "add_estimators": int(add_estimators),
        "new_n_estimators": int(getattr(model, "n_estimators", new_n_estimators)),
        "feature_schema_hash": new_metadata["feature_schema_hash"],
        "rows_labeled": len(labeled),
    }
    manifest_output_path.write_text(json.dumps(_json_safe_value(manifest), indent=2), encoding="utf-8")
    return manifest

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



class EncodedLabelClassifier:
    """Encode string labels for estimators that require numeric labels."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        if sample_weight is not None:
            self.estimator.fit(X, y_encoded, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y_encoded)
        return self

    def predict(self, X):
        pred_encoded = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(pred_encoded.astype(int))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", [])

    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self


def _fit_model(model, X, y):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        return model.fit(X, y)

def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _lightgbm():
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _xgboost():
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    estimator = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return EncodedLabelClassifier(estimator)


def create_model(model_type: str):
    if model_type == "random_forest":
        return _random_forest()
    if model_type == "extra_trees":
        return _extra_trees()
    if model_type == "lightgbm":
        return _lightgbm()
    if model_type == "xgboost":
        return _xgboost()
    raise ValueError(f"Unsupported model_type: {model_type}")


def _atomic_write_model_and_metadata(bundle: dict, metadata: dict, cfg: TradingConfig, model_output_path: str | Path | None = None, metadata_output_path: str | Path | None = None) -> None:
    model_path = Path(model_output_path or cfg.model_path)
    metadata_path = Path(metadata_output_path or cfg.metadata_path)
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
    model_output_path: str | Path | None = None,
    metadata_output_path: str | Path | None = None,
    metadata_extra: dict | None = None,
) -> dict:
    """Train RandomForest from supplied dataframe, optionally saving model artifacts."""
    if "label" in raw_df.columns:
        labeled = raw_df.copy().reset_index(drop=True)
    else:
        featured = build_features(raw_df)
        labeled = add_labels(
            featured,
            cfg.symbol,
            cfg.horizon,
            cfg.pip_threshold,
            cfg.label_method,
            cfg.label_atr_tp_mult,
            cfg.label_atr_sl_mult,
        )

    label_distribution = validate_training_data(labeled)
    cols = feature_cols or feature_columns(labeled)
    validate_no_future_columns(cols)
    x_train, y_train = labeled[cols], labeled["label"]
    validate_features(x_train, cols)

    model = create_model(cfg.model_type)
    _fit_model(model, x_train, y_train)

    metadata = {
        "model_type": cfg.model_type,
        "model_class": type(model).__name__,
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
    if metadata_extra:
        metadata.update(metadata_extra)
    bundle = {"model": model, "features": cols, "config": cfg.__dict__, "metadata": metadata}
    if save_artifacts:
        _atomic_write_model_and_metadata(bundle, metadata, cfg, model_output_path, metadata_output_path)
    return bundle


def train_random_forest(raw_df: pd.DataFrame, cfg: TradingConfig) -> dict:
    featured = build_features(raw_df)
    labeled = add_labels(
        featured,
        cfg.symbol,
        cfg.horizon,
        cfg.pip_threshold,
        cfg.label_method,
        cfg.label_atr_tp_mult,
        cfg.label_atr_sl_mult,
    )
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

    model = create_model(cfg.model_type)
    _fit_model(model, x_train, y_train)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    metadata = {
        "model_type": cfg.model_type,
        "model_class": type(model).__name__,
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

    importances = pd.DataFrame({"feature": cols, "importance": getattr(model, "feature_importances_", [0.0] * len(cols))})
    importances.sort_values("importance", ascending=False).to_csv("reports/feature_importance.csv", index=False)
    return metrics

