"""Backtest with ATR SL, 2R TP, spread/slippage, intrabar high/low, one-position rule."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd

from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL, TradingConfig
from .labels import pip_size
from .signals import generate_signals


DIRECTIONS = {"BOTH", "BUY", "SELL"}


@dataclass(frozen=True)
class BacktestSettings:
    initial_equity: float = 10_000.0
    slippage_pips: float = 0.2
    pip_value_per_lot: float = 10.0


@dataclass(frozen=True)
class SignalFilters:
    min_atr_percentile: float | None = None
    max_atr_percentile: float | None = None
    max_spread_percentile: float | None = None
    max_spread_to_atr: float | None = None
    allowed_sessions: tuple[str, ...] | None = None
    require_bear_stack_for_sell: bool = False
    require_bull_stack_for_buy: bool = False


SESSION_COLUMNS = {
    "asia": "is_asia_session",
    "london": "is_london_session",
    "new_york": "is_new_york_session",
    "ny": "is_new_york_session",
    "rollover": "is_rollover_session",
}


def _filter_value(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if pd.isna(value):
        return None
    return float(value)


def _passes_signal_filters(row: pd.Series, signal: str, filters: SignalFilters | None) -> bool:
    if filters is None:
        return True
    atr_pct = _filter_value(row, "atr_percentile_100")
    if filters.min_atr_percentile is not None and atr_pct is not None and atr_pct < filters.min_atr_percentile:
        return False
    if filters.max_atr_percentile is not None and atr_pct is not None and atr_pct > filters.max_atr_percentile:
        return False
    spread_pct = _filter_value(row, "spread_percentile_100")
    if filters.max_spread_percentile is not None and spread_pct is not None and spread_pct > filters.max_spread_percentile:
        return False
    spread_to_atr = _filter_value(row, "spread_to_atr")
    if filters.max_spread_to_atr is not None and spread_to_atr is not None and spread_to_atr > filters.max_spread_to_atr:
        return False
    if filters.allowed_sessions:
        session_hit = False
        for name in filters.allowed_sessions:
            column = SESSION_COLUMNS.get(name.lower())
            if column and int(row.get(column, 0) or 0) == 1:
                session_hit = True
                break
        if not session_hit:
            return False
    if filters.require_bear_stack_for_sell and signal == LABEL_SELL and int(row.get("trend_stack_bear", 0) or 0) != 1:
        return False
    if filters.require_bull_stack_for_buy and signal == LABEL_BUY and int(row.get("trend_stack_bull", 0) or 0) != 1:
        return False
    return True


def _test_start_time(cfg: TradingConfig) -> pd.Timestamp | None:
    if not os.path.exists(cfg.model_path):
        return None
    bundle = joblib.load(cfg.model_path)
    meta = bundle.get("metadata", {})
    value = meta.get("test_start_time")
    return pd.to_datetime(value) if value else None


def _apply_out_of_sample_filter(signals: pd.DataFrame, cfg: TradingConfig, allow_in_sample: bool) -> pd.DataFrame:
    if allow_in_sample:
        return signals.reset_index(drop=True)
    start = _test_start_time(cfg)
    if start is not None:
        filtered = signals[pd.to_datetime(signals["time"]) >= start].copy()
        return filtered.reset_index(drop=True)
    test_start = int(len(signals) * 0.85)
    return signals.iloc[test_start:].reset_index(drop=True)


def _entry_price(row: pd.Series, signal: str, spread_price: float, slippage_price: float) -> float:
    if signal == LABEL_BUY:
        return float(row["open"] + spread_price + slippage_price)
    return float(row["open"] - slippage_price)


def _exit_price(level: float, signal: str, spread_price: float, slippage_price: float) -> float:
    if signal == LABEL_BUY:
        return float(level - slippage_price)
    return float(level + spread_price + slippage_price)


def _lot_size(equity: float, cfg: TradingConfig, sl_pips: float, pip_value_per_lot: float) -> float:
    risk_amount = equity * cfg.risk_per_trade
    return max(round(risk_amount / max(sl_pips * pip_value_per_lot, 1e-9), 2), 0.01)


def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_losses = 0
    current = 0
    for value in pnl:
        if value <= 0:
            current += 1
            max_losses = max(max_losses, current)
        else:
            current = 0
    return max_losses


def _normalize_direction(direction: str) -> str:
    value = direction.upper()
    if value not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}, got {direction}")
    return value


def _apply_direction_filter(signals: pd.DataFrame, direction: str) -> pd.DataFrame:
    direction = _normalize_direction(direction)
    if direction == "BOTH":
        return signals

    filtered = signals.copy()
    excluded = LABEL_SELL if direction == "BUY" else LABEL_BUY
    filtered.loc[filtered["signal"] == excluded, "signal"] = LABEL_HOLD
    return filtered


def _trade_metrics(trades_df: pd.DataFrame, size: float, initial_equity: float) -> dict:
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_rate": 0,
            "profit_factor": None,
            "expectancy_r": 0,
            "average_r": 0,
            "max_drawdown_pct": 0,
        }

    wins = trades_df[trades_df["pnl_money"] > 0]
    losses = trades_df[trades_df["pnl_money"] <= 0]
    gross_profit = wins["pnl_money"].sum()
    gross_loss = abs(losses["pnl_money"].sum())
    r_multiple = trades_df["pnl_pips"] / (abs(trades_df["entry"] - trades_df["sl"]) / size)
    equity_curve = initial_equity + trades_df["pnl_money"].cumsum()
    drawdown = equity_curve - equity_curve.cummax()

    return {
        "trade_count": int(len(trades_df)),
        "win_rate": float(len(wins) / len(trades_df)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "expectancy_r": float(r_multiple.mean()),
        "average_r": float(r_multiple.mean()),
        "max_drawdown_pct": float(abs(drawdown.min()) / initial_equity * 100),
    }


def _summarize_trades(trades_df: pd.DataFrame, settings: BacktestSettings, size: float, equity: float) -> dict:
    wins = trades_df[trades_df["pnl_money"] > 0]
    losses = trades_df[trades_df["pnl_money"] <= 0]
    gross_profit = wins["pnl_money"].sum()
    gross_loss = abs(losses["pnl_money"].sum())
    equity_series = trades_df["equity"]
    drawdown = equity_series - equity_series.cummax()
    r_multiple = trades_df["pnl_pips"] / (abs(trades_df["entry"] - trades_df["sl"]) / size)
    buy_metrics = _trade_metrics(trades_df[trades_df["signal"] == LABEL_BUY], size, settings.initial_equity)
    sell_metrics = _trade_metrics(trades_df[trades_df["signal"] == LABEL_SELL], size, settings.initial_equity)
    return {
        "trade_count": int(len(trades_df)),
        "win_rate": float(len(wins) / len(trades_df)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "expectancy_money": float(trades_df["pnl_money"].mean()),
        "expectancy_pips": float(trades_df["pnl_pips"].mean()),
        "expectancy_r": float(r_multiple.mean()),
        "average_r": float(r_multiple.mean()),
        "max_drawdown_money": float(drawdown.min()),
        "max_drawdown_pct": float(abs(drawdown.min()) / settings.initial_equity * 100),
        "net_profit_money": float(trades_df["pnl_money"].sum()),
        "net_profit": float(trades_df["pnl_money"].sum()),
        "return_pct": float((equity - settings.initial_equity) / settings.initial_equity * 100),
        "ending_equity": float(equity),
        "final_equity": float(equity),
        "buy_trades": buy_metrics["trade_count"],
        "sell_trades": sell_metrics["trade_count"],
        "buy_win_rate": buy_metrics["win_rate"],
        "sell_win_rate": sell_metrics["win_rate"],
        "buy_profit_factor": buy_metrics["profit_factor"],
        "sell_profit_factor": sell_metrics["profit_factor"],
        "buy_expectancy_r": buy_metrics["expectancy_r"],
        "sell_expectancy_r": sell_metrics["expectancy_r"],
        "buy_average_r": buy_metrics["average_r"],
        "sell_average_r": sell_metrics["average_r"],
        "buy_max_drawdown_pct": buy_metrics["max_drawdown_pct"],
        "sell_max_drawdown_pct": sell_metrics["max_drawdown_pct"],
        "max_consecutive_losses": int(_max_consecutive_losses(trades_df["pnl_money"])),
        "sharpe_like": float(np.sqrt(252) * trades_df["pnl_money"].mean() / trades_df["pnl_money"].std()) if trades_df["pnl_money"].std() else None,
    }


def simulate_signals(
    signals: pd.DataFrame,
    cfg: TradingConfig,
    direction: str = "BOTH",
    probability_threshold: float | None = None,
    filters: SignalFilters | None = None,
    write_reports: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    direction = _normalize_direction(direction)
    settings = BacktestSettings()
    signals = _apply_direction_filter(signals, direction).reset_index(drop=True)
    size = pip_size(cfg.symbol)
    slippage_price = settings.slippage_pips * size
    trades = []
    equity_points = []
    hold_filtered_count = int((signals["signal"] == LABEL_HOLD).sum())
    rule_filtered_count = 0
    equity = settings.initial_equity
    i = 0

    while i < len(signals) - 2:
        row = signals.iloc[i]
        signal = row["signal"]
        if signal not in (LABEL_BUY, LABEL_SELL):
            i += 1
            continue
        if not _passes_signal_filters(row, signal, filters):
            rule_filtered_count += 1
            i += 1
            continue

        entry_row_index = i + 1
        entry_row = signals.iloc[entry_row_index]
        spread_points = float(entry_row.get("spread", 0.0) or 0.0)
        spread_price = spread_points * size
        atr = float(row.get("atr_14", 0.0) or 0.0)
        if atr <= 0:
            i += 1
            continue

        entry = _entry_price(entry_row, signal, spread_price, slippage_price)
        sl_distance = atr * cfg.atr_sl_multiplier
        tp_distance = sl_distance * cfg.reward_risk_ratio
        if signal == LABEL_BUY:
            sl_level = entry - sl_distance
            tp_level = entry + tp_distance
        else:
            sl_level = entry + sl_distance
            tp_level = entry - tp_distance

        sl_pips = abs(entry - sl_level) / size
        lots = _lot_size(equity, cfg, sl_pips, settings.pip_value_per_lot)
        exit_reason = "HORIZON"
        exit_index = min(entry_row_index + cfg.horizon, len(signals) - 1)
        exit_level = float(signals.iloc[exit_index]["close"])

        for j in range(entry_row_index, min(entry_row_index + cfg.horizon + 1, len(signals))):
            candle = signals.iloc[j]
            high = float(candle["high"])
            low = float(candle["low"])
            if signal == LABEL_BUY:
                sl_hit = low <= sl_level
                tp_hit = high >= tp_level
            else:
                sl_hit = high >= sl_level
                tp_hit = low <= tp_level

            if sl_hit and tp_hit:
                exit_reason = "SL_AND_TP_SAME_CANDLE_SL_FIRST"
                exit_level = sl_level
                exit_index = j
                break
            if sl_hit:
                exit_reason = "SL"
                exit_level = sl_level
                exit_index = j
                break
            if tp_hit:
                exit_reason = "TP"
                exit_level = tp_level
                exit_index = j
                break

        exit_fill = _exit_price(exit_level, signal, spread_price, slippage_price)
        direction_multiplier = 1 if signal == LABEL_BUY else -1
        pnl_pips = (exit_fill - entry) / size * direction_multiplier
        pnl_money = pnl_pips * settings.pip_value_per_lot * lots
        equity += pnl_money
        r_multiple = pnl_pips / max(sl_pips, 1e-9)
        equity_points.append({"time": str(signals.iloc[exit_index]["time"]), "equity": equity})
        trades.append({
            "entry_time": str(entry_row["time"]),
            "exit_time": str(signals.iloc[exit_index]["time"]),
            "signal": signal,
            "entry": entry,
            "exit": exit_fill,
            "sl": sl_level,
            "tp": tp_level,
            "exit_reason": exit_reason,
            "lots": lots,
            "pnl_pips": float(pnl_pips),
            "pnl_money": float(pnl_money),
            "r_multiple": float(r_multiple),
            "equity": float(equity),
            "confidence": float(row["confidence"]),
            "buy_prob": float(row.get("buy_prob", 0.0)),
            "sell_prob": float(row.get("sell_prob", 0.0)),
            "hold_prob": float(row.get("hold_prob", 0.0)),
            "spread_points": spread_points,
            "slippage_pips": settings.slippage_pips,
        })
        i = exit_index + 1

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)
    if trades_df.empty:
        summary = {
            "trade_count": 0,
            "message": "No trades passed probability threshold",
            "hold_filtered_count": hold_filtered_count,
            "settings": asdict(settings),
            "direction": direction,
            "probability_threshold": cfg.signal_threshold if probability_threshold is None else probability_threshold,
            "rule_filtered_count": rule_filtered_count,
            "filters": asdict(filters) if filters else None,
        }
    else:
        summary = _summarize_trades(trades_df, settings, size, equity)
        summary.update({
            "hold_filtered_count": hold_filtered_count,
            "settings": asdict(settings),
            "direction": direction,
            "probability_threshold": cfg.signal_threshold if probability_threshold is None else probability_threshold,
            "rule_filtered_count": rule_filtered_count,
            "filters": asdict(filters) if filters else None,
        })

    if write_reports:
        os.makedirs("reports", exist_ok=True)
        if not trades_df.empty:
            trades_df.to_csv("reports/backtest_trades.csv", index=False)
            equity_df.to_csv("reports/equity_curve.csv", index=False)
        with open("reports/backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary, trades_df, equity_df


def run_threshold_sweep(
    raw_df: pd.DataFrame,
    cfg: TradingConfig,
    thresholds: list[float],
    allow_in_sample: bool = False,
    direction: str = "BOTH",
    filters: SignalFilters | None = None,
) -> list[dict]:
    direction = _normalize_direction(direction)
    rows = []
    for threshold in thresholds:
        summary = run_backtest(
            raw_df,
            cfg,
            allow_in_sample=allow_in_sample,
            probability_threshold=threshold,
            direction=direction,
            filters=filters,
        )
        rows.append({
            "threshold": float(threshold),
            "trade_count": int(summary.get("trade_count", 0)),
            "win_rate": summary.get("win_rate", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
            "expectancy_r": summary.get("expectancy_r", 0),
            "average_r": summary.get("average_r", 0),
            "final_equity": summary.get("final_equity", summary.get("ending_equity", 0)),
            "buy_trades": int(summary.get("buy_trades", 0)),
            "sell_trades": int(summary.get("sell_trades", 0)),
            "buy_win_rate": summary.get("buy_win_rate", 0),
            "sell_win_rate": summary.get("sell_win_rate", 0),
            "buy_profit_factor": summary.get("buy_profit_factor"),
            "sell_profit_factor": summary.get("sell_profit_factor"),
            "buy_expectancy_r": summary.get("buy_expectancy_r", 0),
            "sell_expectancy_r": summary.get("sell_expectancy_r", 0),
            "buy_average_r": summary.get("buy_average_r", 0),
            "sell_average_r": summary.get("sell_average_r", 0),
            "buy_max_drawdown_pct": summary.get("buy_max_drawdown_pct", 0),
            "sell_max_drawdown_pct": summary.get("sell_max_drawdown_pct", 0),
            "net_profit": summary.get("net_profit", summary.get("net_profit_money", 0)),
            "return_pct": summary.get("return_pct", 0),
            "max_consecutive_losses": int(summary.get("max_consecutive_losses", 0)),
            "hold_filtered_count": int(summary.get("hold_filtered_count", 0)),
            "rule_filtered_count": int(summary.get("rule_filtered_count", 0)),
        })

    os.makedirs("reports", exist_ok=True)
    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    sweep_df.to_csv("reports/threshold_sweep.csv", index=False)
    with open("reports/threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_backtest(
    raw_df: pd.DataFrame,
    cfg: TradingConfig,
    allow_in_sample: bool = False,
    probability_threshold: float | None = None,
    direction: str = "BOTH",
    filters: SignalFilters | None = None,
) -> dict:
    signals = generate_signals(raw_df, cfg, probability_threshold=probability_threshold)
    signals = _apply_out_of_sample_filter(signals, cfg, allow_in_sample)
    summary, _, _ = simulate_signals(
        signals,
        cfg,
        direction=direction,
        probability_threshold=probability_threshold,
        filters=filters,
        write_reports=True,
    )
    summary["allow_in_sample"] = allow_in_sample
    with open("reports/backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
