"""BUY/SELL/HOLD label creation without leaking future columns into training."""
import pandas as pd
from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL


def pip_size(symbol: str) -> float:
    upper = symbol.upper()
    if "JPY" in upper:
        return 0.01
    if "XAU" in upper or "GOLD" in upper:
        return 0.1
    return 0.0001


def add_labels(df: pd.DataFrame, symbol: str, horizon: int, pip_threshold: float) -> pd.DataFrame:
    data = df.copy()
    size = pip_size(symbol)
    data["future_close"] = data["close"].shift(-horizon)
    data["future_return_pips"] = (data["future_close"] - data["close"]) / size
    data["label"] = LABEL_HOLD
    data.loc[data["future_return_pips"] >= pip_threshold, "label"] = LABEL_BUY
    data.loc[data["future_return_pips"] <= -pip_threshold, "label"] = LABEL_SELL
    return data.dropna().reset_index(drop=True)
