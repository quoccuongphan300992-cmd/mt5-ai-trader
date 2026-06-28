"""Probability based BUY/SELL/HOLD signals."""
import joblib
import pandas as pd

from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL, TradingConfig
from .features import build_features
from .validation import validate_features

OPTIONAL_SIGNAL_COLUMNS = [
    "atr_percentile_100",
    "spread_percentile_100",
    "spread_to_atr",
    "is_asia_session",
    "is_london_session",
    "is_new_york_session",
    "is_rollover_session",
    "trend_stack_bull",
    "trend_stack_bear",
    "price_above_ema200",
    "price_below_ema200",
    "ema_200_slope_20",
    "adx_14",
    "realized_vol_percentile_100",
]


def load_model(path: str):
    return joblib.load(path)


def generate_signals(raw_df: pd.DataFrame, cfg: TradingConfig, probability_threshold: float | None = None) -> pd.DataFrame:
    bundle = load_model(cfg.model_path)
    model = bundle["model"]
    features = bundle["features"]
    data = build_features(raw_df)
    validate_features(data, features)
    probs = model.predict_proba(data[features])
    class_index = {name: idx for idx, name in enumerate(model.classes_)}

    def prob(name: str):
        if name not in class_index:
            return [0.0] * len(data)
        return probs[:, class_index[name]]

    threshold = cfg.signal_threshold if probability_threshold is None else probability_threshold
    out = data[["time", "open", "high", "low", "close", "spread", "atr_14"]].copy()
    out["buy_prob"] = prob(LABEL_BUY)
    out["sell_prob"] = prob(LABEL_SELL)
    out["hold_prob"] = prob(LABEL_HOLD)
    for col in OPTIONAL_SIGNAL_COLUMNS:
        if col in data.columns:
            out[col] = data[col]
    out["signal"] = LABEL_HOLD
    out.loc[out["buy_prob"] >= threshold, "signal"] = LABEL_BUY
    out.loc[out["sell_prob"] >= threshold, "signal"] = LABEL_SELL
    out["confidence"] = out[["buy_prob", "sell_prob", "hold_prob"]].max(axis=1)
    return out
