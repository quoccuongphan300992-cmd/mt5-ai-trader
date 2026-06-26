"""Expanding-window walk-forward validation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import simulate_signals
from .config import LABEL_BUY, LABEL_HOLD, LABEL_SELL, TradingConfig
from .features import build_features, feature_columns
from .labels import add_labels
from .train import train_model_from_dataframe

FOLD_COLUMNS = ["fold", "threshold", "direction", "train_start_time", "train_end_time", "test_start_time", "test_end_time", "train_rows", "test_rows", "trade_count", "win_rate", "profit_factor", "expectancy_r", "average_r", "max_drawdown_pct", "net_profit", "return_pct", "buy_trades", "sell_trades", "status", "error"]
TRADE_COLUMNS = ["fold", "threshold", "direction", "entry_time", "exit_time", "signal", "entry_price", "exit_price", "stop_loss", "take_profit", "exit_reason", "pnl_pips", "pnl_money", "r_multiple", "equity_after", "buy_prob", "sell_prob", "hold_prob", "confidence"]
SUMMARY_COLUMNS = ["threshold", "direction", "folds", "ok_folds", "failed_folds", "total_trades", "positive_expectancy_folds", "positive_pf_folds", "overall_profit_factor", "overall_expectancy_r", "average_fold_expectancy_r", "worst_fold_expectancy_r", "best_fold_expectancy_r", "max_fold_drawdown_pct", "total_net_profit", "average_return_pct", "candidate_pass"]


@dataclass(frozen=True)
class WalkForwardSettings:
    thresholds: list[float]
    direction: str = "SELL"
    folds: int = 5
    initial_train_pct: float = 0.50
    test_pct: float = 0.10


def build_thresholds(threshold: float | None, min_value: float, max_value: float, step: float) -> list[float]:
    if threshold is not None:
        return [round(float(threshold), 4)]
    values = []
    value = min_value
    while value <= max_value + 1e-12:
        values.append(round(value, 4))
        value += step
    return values


def make_walk_forward_splits(n_rows: int, folds: int = 5, initial_train_pct: float = 0.50, test_pct: float = 0.10) -> list[tuple[int, int, int]]:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    splits = []
    for fold_idx in range(folds):
        train_end = int(n_rows * (initial_train_pct + fold_idx * test_pct))
        test_end = int(n_rows * (initial_train_pct + (fold_idx + 1) * test_pct))
        if fold_idx == folds - 1:
            test_end = n_rows
        if train_end <= 0 or test_end <= train_end:
            continue
        if test_end > n_rows:
            break
        splits.append((0, train_end, test_end))
    return splits


def _probability_columns(bundle: dict, test_df: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    model = bundle["model"]
    probs = model.predict_proba(test_df[features])
    class_index = {name: idx for idx, name in enumerate(model.classes_)}

    def prob(name: str):
        if name not in class_index:
            return np.zeros(len(test_df))
        return probs[:, class_index[name]]

    out = test_df[["time", "open", "high", "low", "close", "spread", "atr_14"]].copy()
    out["buy_prob"] = prob(LABEL_BUY)
    out["sell_prob"] = prob(LABEL_SELL)
    out["hold_prob"] = prob(LABEL_HOLD)
    out["signal"] = LABEL_HOLD
    out.loc[out["buy_prob"] >= threshold, "signal"] = LABEL_BUY
    out.loc[out["sell_prob"] >= threshold, "signal"] = LABEL_SELL
    out["confidence"] = out[["buy_prob", "sell_prob", "hold_prob"]].max(axis=1)
    return out


def _fold_row(fold, threshold, direction, train_df, test_df, status, error="", summary=None):
    summary = summary or {}
    return {
        "fold": fold, "threshold": threshold, "direction": direction,
        "train_start_time": str(train_df["time"].iloc[0]) if not train_df.empty else "",
        "train_end_time": str(train_df["time"].iloc[-1]) if not train_df.empty else "",
        "test_start_time": str(test_df["time"].iloc[0]) if not test_df.empty else "",
        "test_end_time": str(test_df["time"].iloc[-1]) if not test_df.empty else "",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "trade_count": int(summary.get("trade_count", 0)), "win_rate": summary.get("win_rate", 0),
        "profit_factor": summary.get("profit_factor"), "expectancy_r": summary.get("expectancy_r", 0),
        "average_r": summary.get("average_r", 0), "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
        "net_profit": summary.get("net_profit", 0), "return_pct": summary.get("return_pct", 0),
        "buy_trades": int(summary.get("buy_trades", 0)), "sell_trades": int(summary.get("sell_trades", 0)),
        "status": status, "error": error,
    }


def _format_trades(trades_df, fold, threshold, direction):
    if trades_df.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    out = pd.DataFrame({
        "fold": fold, "threshold": threshold, "direction": direction,
        "entry_time": trades_df["entry_time"], "exit_time": trades_df["exit_time"], "signal": trades_df["signal"],
        "entry_price": trades_df["entry"], "exit_price": trades_df["exit"], "stop_loss": trades_df["sl"],
        "take_profit": trades_df["tp"], "exit_reason": trades_df["exit_reason"], "pnl_pips": trades_df["pnl_pips"],
        "pnl_money": trades_df["pnl_money"], "r_multiple": trades_df["r_multiple"], "equity_after": trades_df["equity"],
        "buy_prob": trades_df["buy_prob"], "sell_prob": trades_df["sell_prob"], "hold_prob": trades_df["hold_prob"],
        "confidence": trades_df["confidence"],
    })
    return out[TRADE_COLUMNS]


def _summary_for_threshold(threshold, direction, folds_df, trades_df):
    threshold_folds = folds_df[folds_df["threshold"] == threshold]
    ok_folds = threshold_folds[threshold_folds["status"] == "ok"]
    threshold_trades = trades_df[trades_df["threshold"] == threshold] if not trades_df.empty else pd.DataFrame()
    if threshold_trades.empty:
        overall_profit_factor = None; overall_expectancy_r = 0.0; total_net_profit = 0.0
    else:
        wins = threshold_trades[threshold_trades["pnl_money"] > 0]
        losses = threshold_trades[threshold_trades["pnl_money"] <= 0]
        gross_loss = float(abs(losses["pnl_money"].sum()))
        overall_profit_factor = float(wins["pnl_money"].sum()) / gross_loss if gross_loss else None
        overall_expectancy_r = float(threshold_trades["r_multiple"].mean())
        total_net_profit = float(threshold_trades["pnl_money"].sum())
    positive_expectancy_folds = int((ok_folds["expectancy_r"] > 0).sum()) if not ok_folds.empty else 0
    positive_pf_folds = int((ok_folds["profit_factor"].fillna(0) > 1).sum()) if not ok_folds.empty else 0
    max_fold_drawdown_pct = float(ok_folds["max_drawdown_pct"].max()) if not ok_folds.empty else 0.0
    total_trades = int(ok_folds["trade_count"].sum()) if not ok_folds.empty else 0
    candidate_pass = total_trades >= 60 and positive_expectancy_folds >= 3 and positive_pf_folds >= 3 and overall_profit_factor is not None and overall_profit_factor > 1.05 and overall_expectancy_r > 0 and max_fold_drawdown_pct < 20
    return {"threshold": threshold, "direction": direction, "folds": int(len(threshold_folds)), "ok_folds": int(len(ok_folds)), "failed_folds": int((threshold_folds["status"] != "ok").sum()), "total_trades": total_trades, "positive_expectancy_folds": positive_expectancy_folds, "positive_pf_folds": positive_pf_folds, "overall_profit_factor": overall_profit_factor, "overall_expectancy_r": overall_expectancy_r, "average_fold_expectancy_r": float(ok_folds["expectancy_r"].mean()) if not ok_folds.empty else 0.0, "worst_fold_expectancy_r": float(ok_folds["expectancy_r"].min()) if not ok_folds.empty else 0.0, "best_fold_expectancy_r": float(ok_folds["expectancy_r"].max()) if not ok_folds.empty else 0.0, "max_fold_drawdown_pct": max_fold_drawdown_pct, "total_net_profit": total_net_profit, "average_return_pct": float(ok_folds["return_pct"].mean()) if not ok_folds.empty else 0.0, "candidate_pass": bool(candidate_pass)}


def run_walk_forward(raw_df: pd.DataFrame, cfg: TradingConfig, settings: WalkForwardSettings) -> list[dict]:
    os.makedirs("reports", exist_ok=True)
    labeled = add_labels(build_features(raw_df), cfg.symbol, cfg.horizon, cfg.pip_threshold)
    features = feature_columns(labeled)
    splits = make_walk_forward_splits(len(labeled), settings.folds, settings.initial_train_pct, settings.test_pct)
    fold_rows = []
    trade_frames = []
    for threshold in settings.thresholds:
        for fold_no, (train_start, train_end, test_end) in enumerate(splits, start=1):
            train_df = labeled.iloc[train_start:train_end].reset_index(drop=True)
            test_df = labeled.iloc[train_end:test_end].reset_index(drop=True)
            try:
                bundle = train_model_from_dataframe(train_df, cfg, feature_cols=features, save_artifacts=False)
                signals = _probability_columns(bundle, test_df, features, threshold)
                summary, trades_df, _ = simulate_signals(signals, cfg, direction=settings.direction, probability_threshold=threshold, write_reports=False)
                fold_rows.append(_fold_row(fold_no, threshold, settings.direction, train_df, test_df, "ok", summary=summary))
                formatted = _format_trades(trades_df, fold_no, threshold, settings.direction)
                if not formatted.empty:
                    trade_frames.append(formatted)
            except Exception as exc:
                fold_rows.append(_fold_row(fold_no, threshold, settings.direction, train_df, test_df, "failed", error=str(exc)))
    folds_df = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=TRADE_COLUMNS)
    summaries = [_summary_for_threshold(threshold, settings.direction, folds_df, trades_df) for threshold in settings.thresholds]
    pd.DataFrame(summaries, columns=SUMMARY_COLUMNS).to_csv("reports/walk_forward_summary.csv", index=False)
    folds_df.to_csv("reports/walk_forward_folds.csv", index=False)
    trades_df.to_csv("reports/walk_forward_trades.csv", index=False)
    with open("reports/walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    return summaries
