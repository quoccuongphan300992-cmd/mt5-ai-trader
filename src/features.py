"""Feature engineering for OHLCV forex candles."""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("time").reset_index(drop=True)

    close = data["close"]
    high = data["high"]
    low = data["low"]
    open_ = data["open"]

    data["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    data["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    data["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
    data["ema_20_50_dist"] = data["ema_20"] - data["ema_50"]
    data["ema_50_200_dist"] = data["ema_50"] - data["ema_200"]
    data["close_ema20_dist"] = close - data["ema_20"]
    data["close_ema50_dist"] = close - data["ema_50"]

    data["rsi_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()
    stoch = StochasticOscillator(high, low, close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    atr = AverageTrueRange(high, low, close, window=14)
    data["atr_14"] = atr.average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_width"] = data["bb_high"] - data["bb_low"]
    data["rolling_std_20"] = close.rolling(20).std()

    data["body"] = close - open_
    data["candle_range"] = (high - low).replace(0, np.nan)
    data["body_pct"] = data["body"] / data["candle_range"]
    data["upper_wick"] = high - np.maximum(open_, close)
    data["lower_wick"] = np.minimum(open_, close) - low
    data["gap"] = open_ - close.shift(1)
    data["close_position_in_range"] = (close - low) / data["candle_range"]

    data["return_1"] = close.pct_change(1)
    data["return_5"] = close.pct_change(5)
    data["return_20"] = close.pct_change(20)
    mean_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    data["zscore_20"] = (close - mean_20) / std_20

    if "spread" in data.columns:
        data["spread"] = data["spread"].fillna(0)

    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"time", "label", "future_close", "future_return_pips"}
    return [c for c in df.columns if c not in excluded]
