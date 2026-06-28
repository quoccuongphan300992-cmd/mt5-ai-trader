"""CLI for MT5 supervised ML forex model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import SignalFilters, run_backtest, run_threshold_sweep
from src.config import TradingConfig
from src.data import latest_raw_csv, load_csv
from src.mt5_client import fetch_rates, initialize_mt5, save_rates, shutdown_mt5
from src.signals import generate_signals
from src.train import train_random_forest
from src.walk_forward import WalkForwardSettings, build_thresholds, run_walk_forward


def sample_data(rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    time = pd.date_range("2020-01-01", periods=rows, freq="h")
    returns = rng.normal(0, 0.0007, rows)
    close = 1.10 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.integers(8, 20, rows)
    high = np.maximum(open_, close) + rng.uniform(0.00005, 0.0008, rows)
    low = np.minimum(open_, close) - rng.uniform(0.00005, 0.0008, rows)
    return pd.DataFrame({
        "time": time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": rng.integers(100, 2000, rows),
        "spread": spread,
        "real_volume": 0,
    })


def resolve_data(path: str | None, sample: bool) -> pd.DataFrame:
    if sample:
        return sample_data()
    chosen = path or latest_raw_csv()
    if not chosen:
        raise SystemExit("No CSV found. Run fetch first or use --sample.")
    return load_csv(chosen)


def build_signal_filters(args) -> SignalFilters:
    sessions = tuple(s.strip().lower() for s in args.sessions.split(",") if s.strip()) if getattr(args, "sessions", "") else None
    return SignalFilters(
        min_atr_percentile=getattr(args, "min_atr_percentile", None),
        max_atr_percentile=getattr(args, "max_atr_percentile", None),
        max_spread_percentile=getattr(args, "max_spread_percentile", None),
        max_spread_to_atr=getattr(args, "max_spread_to_atr", None),
        allowed_sessions=sessions,
        require_bear_stack_for_sell=getattr(args, "require_bear_stack_for_sell", False),
        require_bull_stack_for_buy=getattr(args, "require_bull_stack_for_buy", False),
    )


def add_filter_args(p):
    p.add_argument("--min-atr-percentile", type=float)
    p.add_argument("--max-atr-percentile", type=float)
    p.add_argument("--max-spread-percentile", type=float)
    p.add_argument("--max-spread-to-atr", type=float)
    p.add_argument("--sessions", default="", help="Comma list: asia,london,new_york,rollover")
    p.add_argument("--require-bear-stack-for-sell", action="store_true")
    p.add_argument("--require-bull-stack-for-buy", action="store_true")


def build_config(args) -> TradingConfig:
    return TradingConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        horizon=args.horizon,
        pip_threshold=getattr(args, "pip_threshold", 30.0),
        signal_threshold=getattr(args, "signal_threshold", 0.75),
        risk_per_trade=getattr(args, "risk", 0.01),
        trade_mode=getattr(args, "trade_mode", "paper"),
        label_method=getattr(args, "label_method", "fixed_return"),
        label_atr_tp_mult=getattr(args, "label_atr_tp_mult", 1.5),
        label_atr_sl_mult=getattr(args, "label_atr_sl_mult", 1.0),
        model_type=getattr(args, "model_type", "random_forest"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MT5 supervised ML forex model: RandomForest BUY/SELL/HOLD probabilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--symbol", default="EURUSD")
        p.add_argument("--timeframe", default="H1")
        p.add_argument("--bars", type=int, default=100000)
        p.add_argument("--horizon", type=int, default=10)
        p.add_argument("--pip-threshold", type=float, default=30.0)
        p.add_argument("--signal-threshold", type=float, default=0.75)
        p.add_argument("--risk", type=float, default=0.01)
        p.add_argument("--trade-mode", default="paper", choices=["paper", "demo", "live"])
        p.add_argument("--label-method", default="fixed_return", choices=["fixed_return", "atr_path"])
        p.add_argument("--label-atr-tp-mult", type=float, default=1.5)
        p.add_argument("--label-atr-sl-mult", type=float, default=1.0)
        p.add_argument("--model-type", default="random_forest", choices=["random_forest", "extra_trees", "lightgbm", "xgboost"])

    fetch_p = sub.add_parser("fetch", help="Fetch OHLCV data from MT5")
    add_common(fetch_p)

    train_p = sub.add_parser("train", help="Train RandomForest model")
    add_common(train_p)
    train_p.add_argument("--csv")
    train_p.add_argument("--sample", action="store_true")

    backtest_p = sub.add_parser("backtest", help="Backtest probability-filtered signals")
    add_common(backtest_p)
    backtest_p.add_argument("--csv")
    backtest_p.add_argument("--sample", action="store_true")
    backtest_p.add_argument("--allow-in-sample", action="store_true")
    backtest_p.add_argument("--direction", default="BOTH", choices=["BOTH", "BUY", "SELL"])
    add_filter_args(backtest_p)

    signal_p = sub.add_parser("signal", help="Generate latest BUY/SELL/HOLD probability signal")
    add_common(signal_p)
    signal_p.add_argument("--csv")
    signal_p.add_argument("--sample", action="store_true")

    pipeline_p = sub.add_parser("pipeline", help="Run safe fetch -> train -> backtest -> signal pipeline")
    add_common(pipeline_p)
    pipeline_p.add_argument("--csv")
    pipeline_p.add_argument("--sample", action="store_true")
    pipeline_p.add_argument("--allow-in-sample", action="store_true")

    sweep_p = sub.add_parser("threshold-sweep", help="Backtest multiple probability thresholds on the holdout set")
    add_common(sweep_p)
    sweep_p.add_argument("--csv")
    sweep_p.add_argument("--sample", action="store_true")
    sweep_p.add_argument("--allow-in-sample", action="store_true")
    sweep_p.add_argument("--direction", default="BOTH", choices=["BOTH", "BUY", "SELL"])
    sweep_p.add_argument("--min", type=float, default=0.50)
    sweep_p.add_argument("--max", type=float, default=0.90)
    sweep_p.add_argument("--step", type=float, default=0.05)
    add_filter_args(sweep_p)

    walk_p = sub.add_parser("walk-forward", help="Run expanding-window offline walk-forward validation")
    add_common(walk_p)
    walk_p.add_argument("--csv")
    walk_p.add_argument("--sample", action="store_true")
    walk_p.add_argument("--direction", default="SELL", choices=["BOTH", "BUY", "SELL"])
    walk_p.add_argument("--folds", type=int, default=5)
    walk_p.add_argument("--initial-train-pct", type=float, default=0.50)
    walk_p.add_argument("--test-pct", type=float, default=0.10)
    walk_p.add_argument("--threshold", type=float)
    walk_p.add_argument("--min", type=float, default=0.46)
    walk_p.add_argument("--max", type=float, default=0.52)
    walk_p.add_argument("--step", type=float, default=0.01)
    add_filter_args(walk_p)

    auto_p = sub.add_parser("auto-improve", help="Run offline auto-improve model config search using walk-forward validation.")
    auto_p.add_argument("--csv")
    auto_p.add_argument("--symbol", default="EURUSD")
    auto_p.add_argument("--timeframe", default="H1")
    auto_p.add_argument("--bars", type=int, default=100000)
    auto_p.add_argument("--sample", action="store_true")
    auto_p.add_argument("--horizon", type=int, default=None)
    auto_p.add_argument("--pip-threshold", type=float, default=30.0)
    auto_p.add_argument("--signal-threshold", type=float, default=0.75)
    auto_p.add_argument("--risk", type=float, default=0.01)
    auto_p.add_argument("--trade-mode", default="paper", choices=["paper", "demo", "live"])
    auto_p.add_argument("--max-rounds", type=int, default=30)
    auto_p.add_argument("--include-heavy-models", action="store_true", help="Include LightGBM and XGBoost in auto-improve grid.")
    auto_p.add_argument("--min", type=float, default=0.46)
    auto_p.add_argument("--max", type=float, default=0.60)
    auto_p.add_argument("--step", type=float, default=0.01)
    auto_p.add_argument("--folds", type=int, default=5)
    auto_p.add_argument("--initial-train-pct", type=float, default=0.50)
    auto_p.add_argument("--test-pct", type=float, default=0.10)
    auto_p.add_argument("--min-trades", type=int, default=30)
    auto_p.add_argument("--min-profit-factor", type=float, default=1.20)
    auto_p.add_argument("--min-expectancy", type=float, default=0.0)
    auto_p.add_argument("--min-positive-fold-ratio", type=float, default=0.60)
    auto_p.add_argument("--max-drawdown-limit", type=float, default=0.20)
    auto_p.add_argument("--promotion-mode", choices=["candidate-only", "auto-promote"], default="candidate-only")
    auto_p.add_argument("--candidate-model-dir", default="models/candidates")
    auto_p.add_argument("--min-pf-improvement", type=float, default=0.0)
    auto_p.add_argument("--min-trade-improvement", type=int, default=0)
    auto_p.add_argument("--filter-preset", default="grid", choices=["grid", "none", "trend_ema200", "atr_mid", "london_ny", "adx_trend", "avoid_chop", "trend_atr_combo", "spread_safe"], help="Optional filter preset override. Default grid searches priority/expansion presets.")
    add_filter_args(auto_p)

    cont_p = sub.add_parser("continue-train-candidate", help="Continue training an existing auto-improve candidate by adding ensemble trees.")
    cont_p.add_argument("--csv")
    cont_p.add_argument("--sample", action="store_true")
    cont_p.add_argument("--symbol", default="EURUSD")
    cont_p.add_argument("--timeframe", default="H1")
    cont_p.add_argument("--bars", type=int, default=100000)
    cont_p.add_argument("--candidate-id", required=True)
    cont_p.add_argument("--candidate-model-dir", default="models/candidates")
    cont_p.add_argument("--candidate-model-path")
    cont_p.add_argument("--candidate-metadata-path")
    cont_p.add_argument("--add-estimators", type=int, default=300)
    cont_p.add_argument("--output-dir", default="models/candidates")
    cont_p.add_argument("--allow-retrain-fallback", action="store_true")

    args = parser.parse_args()
    cfg = build_config(args)

    if args.command == "fetch":
        initialize_mt5()
        try:
            df = fetch_rates(cfg.symbol, cfg.timeframe, cfg.bars)
            path = save_rates(df, cfg.symbol, cfg.timeframe)
            print(f"Saved {len(df)} rows to {path}")
        finally:
            shutdown_mt5()
        return

    if args.command == "continue-train-candidate":
        from src.auto_improve import continue_train_candidate
        result = continue_train_candidate(args)
        print(json.dumps(result, indent=2))
        return

    if args.command == "train":
        df = resolve_data(args.csv, args.sample)
        metrics = train_random_forest(df, cfg)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "backtest":
        df = resolve_data(args.csv, args.sample)
        summary = run_backtest(df, cfg, allow_in_sample=args.allow_in_sample, direction=args.direction, filters=build_signal_filters(args))
        print(json.dumps(summary, indent=2))
        return

    if args.command == "signal":
        df = resolve_data(args.csv, args.sample)
        signals = generate_signals(df, cfg)
        print(signals.tail(1).to_json(orient="records", indent=2, date_format="iso"))
        return

    if args.command == "pipeline":
        if args.sample:
            print("=== Using sample data ===")
            df = sample_data()
        elif args.csv:
            print(f"=== Loading CSV: {args.csv} ===")
            df = load_csv(args.csv)
        else:
            print("=== Fetch MT5 data ===")
            initialize_mt5()
            try:
                df = fetch_rates(cfg.symbol, cfg.timeframe, cfg.bars)
                path = save_rates(df, cfg.symbol, cfg.timeframe)
                print(f"Saved {len(df)} rows to {path}")
            finally:
                shutdown_mt5()

        print("=== Train model ===")
        metrics = train_random_forest(df, cfg)
        print(json.dumps(metrics, indent=2))

        print("=== Backtest ===")
        summary = run_backtest(df, cfg, allow_in_sample=args.allow_in_sample)
        print(json.dumps(summary, indent=2))

        print("=== Current signal ===")
        signals = generate_signals(df, cfg)
        print(signals.tail(1).to_json(orient="records", indent=2, date_format="iso"))

        print("=== Pipeline completed ===")
        return

    if args.command == "threshold-sweep":
        df = resolve_data(args.csv, args.sample)
        thresholds = []
        value = args.min
        while value <= args.max + 1e-9:
            thresholds.append(round(value, 2))
            value += args.step
        rows = run_threshold_sweep(
            df,
            cfg,
            thresholds,
            allow_in_sample=args.allow_in_sample,
            direction=args.direction,
            filters=build_signal_filters(args),
        )
        print(json.dumps(rows, indent=2))
        print("Saved reports/threshold_sweep.csv and reports/threshold_sweep.json")
        return

    if args.command == "walk-forward":
        df = resolve_data(args.csv, args.sample)
        thresholds = build_thresholds(args.threshold, args.min, args.max, args.step)
        settings = WalkForwardSettings(
            thresholds=thresholds,
            direction=args.direction,
            folds=args.folds,
            initial_train_pct=args.initial_train_pct,
            test_pct=args.test_pct,
            filters=build_signal_filters(args),
        )
        rows = run_walk_forward(df, cfg, settings)
        print(json.dumps(rows, indent=2))
        print("Saved reports/walk_forward_summary.csv, reports/walk_forward_folds.csv, reports/walk_forward_trades.csv, and reports/walk_forward_summary.json")
        return

    if args.command == "auto-improve":
        from src.auto_improve import run_auto_improve

        result = run_auto_improve(args)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
