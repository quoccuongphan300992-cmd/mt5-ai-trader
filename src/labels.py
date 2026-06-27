"""BUY/SELL/HOLD label creation without leaking future columns into training."""
from __future__ import annotations

import pandas as pd

from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL


def pip_size(symbol: str) -> float:
    upper = symbol.upper()
    if "JPY" in upper:
        return 0.01
    if "XAU" in upper or "GOLD" in upper:
        return 0.1
    return 0.0001


def _add_fixed_return_labels(df: pd.DataFrame, symbol: str, horizon: int, pip_threshold: float) -> pd.DataFrame:
    data = df.copy()
    size = pip_size(symbol)
    data["future_close"] = data["close"].shift(-horizon)
    data["future_return_pips"] = (data["future_close"] - data["close"]) / size
    data["label"] = LABEL_HOLD
    data.loc[data["future_return_pips"] >= pip_threshold, "label"] = LABEL_BUY
    data.loc[data["future_return_pips"] <= -pip_threshold, "label"] = LABEL_SELL
    return data.dropna().reset_index(drop=True)


def _resolve_atr_path_label(data: pd.DataFrame, index: int, horizon: int, tp_mult: float, sl_mult: float) -> str:
    atr = float(data.at[index, "atr_14"]) if "atr_14" in data.columns else 0.0
    if pd.isna(atr) or atr <= 0 or index + horizon >= len(data):
        return LABEL_HOLD

    entry = float(data.at[index, "close"])
    buy_tp = entry + atr * tp_mult
    buy_sl = entry - atr * sl_mult
    sell_tp = entry - atr * tp_mult
    sell_sl = entry + atr * sl_mult

    for future_index in range(index + 1, index + horizon + 1):
        high = float(data.at[future_index, "high"])
        low = float(data.at[future_index, "low"])

        buy_tp_hit = high >= buy_tp
        buy_sl_hit = low <= buy_sl
        sell_tp_hit = low <= sell_tp
        sell_sl_hit = high >= sell_sl

        if (buy_tp_hit and buy_sl_hit) or (sell_tp_hit and sell_sl_hit):
            return LABEL_HOLD

        clean_buy = buy_tp_hit and not buy_sl_hit
        clean_sell = sell_tp_hit and not sell_sl_hit
        if clean_buy and clean_sell:
            return LABEL_HOLD
        if clean_buy:
            return LABEL_BUY
        if clean_sell:
            return LABEL_SELL
        if buy_sl_hit or sell_sl_hit:
            return LABEL_HOLD

    return LABEL_HOLD


def _add_atr_path_labels(df: pd.DataFrame, horizon: int, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    data = df.copy().reset_index(drop=True)
    data["future_close"] = data["close"].shift(-horizon)
    data["future_return_pips"] = pd.NA
    data["label"] = [
        _resolve_atr_path_label(data, i, horizon, tp_mult, sl_mult)
        for i in range(len(data))
    ]
    return data.dropna(subset=["future_close"]).reset_index(drop=True)


def add_labels(
    df: pd.DataFrame,
    symbol: str,
    horizon: int,
    pip_threshold: float,
    label_method: str = "fixed_return",
    label_atr_tp_mult: float = 1.5,
    label_atr_sl_mult: float = 1.0,
) -> pd.DataFrame:
    if label_method == "fixed_return":
        return _add_fixed_return_labels(df, symbol, horizon, pip_threshold)
    if label_method == "atr_path":
        return _add_atr_path_labels(df, horizon, label_atr_tp_mult, label_atr_sl_mult)
    raise ValueError(f"Unsupported label_method: {label_method}")
