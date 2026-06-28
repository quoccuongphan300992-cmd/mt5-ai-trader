"""Feature engineering for OHLCV forex candles."""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("time").reset_index(drop=True)

    close = data["close"]
    high = data["high"]
    low = data["low"]
    open_ = data["open"]

    data["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    data["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    data["ema_100"] = EMAIndicator(close, window=100).ema_indicator()
    data["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
    data["ema_20_50_dist"] = data["ema_20"] - data["ema_50"]
    data["ema_50_100_dist"] = data["ema_50"] - data["ema_100"]
    data["ema_50_200_dist"] = data["ema_50"] - data["ema_200"]
    data["close_ema20_dist"] = close - data["ema_20"]
    data["close_ema50_dist"] = close - data["ema_50"]
    data["close_ema100_dist"] = close - data["ema_100"]
    data["close_ema200_dist"] = close - data["ema_200"]
    data["ema_20_slope_5"] = data["ema_20"] - data["ema_20"].shift(5)
    data["ema_50_slope_10"] = data["ema_50"] - data["ema_50"].shift(10)
    data["ema_100_slope_20"] = data["ema_100"] - data["ema_100"].shift(20)
    data["ema_200_slope_20"] = data["ema_200"] - data["ema_200"].shift(20)
    data["ema_20_slope_5_pct"] = data["ema_20_slope_5"] / data["ema_20"].replace(0, np.nan)
    data["ema_50_slope_10_pct"] = data["ema_50_slope_10"] / data["ema_50"].replace(0, np.nan)
    data["ema_200_slope_20_pct"] = data["ema_200_slope_20"] / data["ema_200"].replace(0, np.nan)
    data["ema_20_above_50"] = (data["ema_20"] > data["ema_50"]).astype(int)
    data["ema_50_above_100"] = (data["ema_50"] > data["ema_100"]).astype(int)
    data["ema_50_above_200"] = (data["ema_50"] > data["ema_200"]).astype(int)
    data["price_above_ema200"] = (close > data["ema_200"]).astype(int)
    data["price_below_ema200"] = (close < data["ema_200"]).astype(int)
    data["trend_stack_bull"] = ((data["ema_20"] > data["ema_50"]) & (data["ema_50"] > data["ema_200"])).astype(int)
    data["trend_stack_bear"] = ((data["ema_20"] < data["ema_50"]) & (data["ema_50"] < data["ema_200"])).astype(int)

    data["rsi_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()
    stoch = StochasticOscillator(high, low, close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()
    adx = ADXIndicator(high, low, close, window=14)
    data["adx_14"] = adx.adx()
    data["adx_pos_14"] = adx.adx_pos()
    data["adx_neg_14"] = adx.adx_neg()

    atr = AverageTrueRange(high, low, close, window=14)
    data["atr_14"] = atr.average_true_range()
    data["atr_ma_50"] = data["atr_14"].rolling(50).mean()
    data["atr_ratio_50"] = data["atr_14"] / data["atr_ma_50"].replace(0, np.nan)
    data["atr_pct_close"] = data["atr_14"] / close.replace(0, np.nan)
    data["atr_percentile_100"] = rolling_percentile_rank(data["atr_14"], 100)
    bb = BollingerBands(close, window=20, window_dev=2)
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_width"] = data["bb_high"] - data["bb_low"]
    data["bb_width_pct_close"] = data["bb_width"] / close.replace(0, np.nan)
    data["rolling_std_20"] = close.rolling(20).std()
    data["rolling_std_50"] = close.rolling(50).std()
    data["rolling_std_percentile_100"] = rolling_percentile_rank(data["rolling_std_20"], 100)
    data["realized_vol_20"] = close.pct_change().rolling(20).std()
    data["realized_vol_50"] = close.pct_change().rolling(50).std()
    data["realized_vol_percentile_100"] = rolling_percentile_rank(data["realized_vol_20"], 100)

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

    data["time"] = pd.to_datetime(data["time"])
    data["hour"] = data["time"].dt.hour
    data["day_of_week"] = data["time"].dt.dayofweek
    data["is_asia_session"] = ((data["hour"] >= 0) & (data["hour"] < 7)).astype(int)
    data["is_london_session"] = ((data["hour"] >= 7) & (data["hour"] < 13)).astype(int)
    data["is_new_york_session"] = ((data["hour"] >= 13) & (data["hour"] < 21)).astype(int)
    data["is_rollover_session"] = ((data["hour"] >= 21) & (data["hour"] < 24)).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    if "spread" in data.columns:
        data["spread"] = data["spread"].fillna(0)
        data["spread_percentile_100"] = rolling_percentile_rank(data["spread"], 100)
        data["spread_to_atr"] = data["spread"] / data["atr_14"].replace(0, np.nan)
    if "tick_volume" in data.columns:
        data["tick_volume_percentile_100"] = rolling_percentile_rank(data["tick_volume"], 100)

    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"time", "label", "future_close", "future_return_pips"}
    return [c for c in df.columns if c not in excluded]
