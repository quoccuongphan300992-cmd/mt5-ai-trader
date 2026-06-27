"""Default configuration for supervised ML forex pipeline."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    bars: int = 100_000
    horizon: int = 10
    pip_threshold: float = 30.0
    signal_threshold: float = 0.75
    risk_per_trade: float = 0.01
    atr_period: int = 14
    atr_sl_multiplier: float = 1.0
    reward_risk_ratio: float = 2.0
    trade_mode: str = "paper"
    label_method: str = "fixed_return"
    label_atr_tp_mult: float = 1.5
    label_atr_sl_mult: float = 1.0
    model_type: str = "random_forest"
    model_path: str = "models/model.joblib"
    metadata_path: str = "models/metadata.json"


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


LABEL_BUY = "BUY"
LABEL_SELL = "SELL"
LABEL_HOLD = "HOLD"
LABELS = [LABEL_BUY, LABEL_SELL, LABEL_HOLD]
