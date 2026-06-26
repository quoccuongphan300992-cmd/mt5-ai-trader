"""Risk engine: model predicts signal, this module sizes trades."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPlan:
    risk_amount: float
    sl_distance: float
    tp_distance: float
    lot_size: float


def calculate_risk_plan(
    equity: float,
    risk_per_trade: float,
    atr: float,
    atr_sl_multiplier: float,
    reward_risk_ratio: float,
    pip_value_per_lot: float = 10.0,
    pip_size: float = 0.0001,
) -> RiskPlan:
    risk_amount = equity * risk_per_trade
    sl_distance = atr * atr_sl_multiplier
    tp_distance = sl_distance * reward_risk_ratio
    sl_pips = max(sl_distance / pip_size, 1e-9)
    lot_size = risk_amount / (sl_pips * pip_value_per_lot)
    return RiskPlan(risk_amount, sl_distance, tp_distance, round(lot_size, 2))
