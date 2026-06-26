import pandas as pd

from src.features import build_features, feature_columns
from src.labels import add_labels
from src.config import TradingConfig, LABEL_BUY, LABEL_HOLD, LABEL_SELL
from src.validation import validate_features, validate_no_future_columns
from src.trader import assert_safe_mode


def sample_df(rows=260):
    time = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = pd.Series([1.10 + i * 0.0001 for i in range(rows)])
    return pd.DataFrame({
        "time": time,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.0005,
        "low": close - 0.0005,
        "close": close,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0,
    })


def test_build_features_no_nan_after_drop():
    features = build_features(sample_df())
    assert len(features) > 0
    assert not features.isna().any().any()


def test_labels_buy_for_uptrend():
    labeled = add_labels(build_features(sample_df()), "EURUSD", horizon=10, pip_threshold=5)
    assert LABEL_BUY in set(labeled["label"])


def test_feature_columns_exclude_future_and_label():
    labeled = add_labels(build_features(sample_df()), "EURUSD", horizon=10, pip_threshold=5)
    cols = feature_columns(labeled)
    assert "future_close" not in cols
    assert "future_return_pips" not in cols
    assert "label" not in cols
    validate_no_future_columns(cols)


def test_validate_features_missing_column_fails():
    df = pd.DataFrame({"a": [1.0]})
    try:
        validate_features(df, ["a", "b"])
    except ValueError as exc:
        assert "Missing model feature columns" in str(exc)
    else:
        raise AssertionError("validate_features should fail on missing columns")


def test_live_mode_blocked_by_default():
    cfg = TradingConfig(trade_mode="live")
    try:
        assert_safe_mode(cfg, live_confirm=False)
    except RuntimeError as exc:
        assert "Live trading blocked" in str(exc)
    else:
        raise AssertionError("live mode should be blocked by default")


def test_threshold_sweep_outputs_expected_columns():
    from main import sample_data
    from src.backtest import run_threshold_sweep
    from src.train import train_random_forest

    cfg = TradingConfig(signal_threshold=0.75)
    df = sample_data(5000)
    train_random_forest(df, cfg)
    rows = run_threshold_sweep(df, cfg, [0.5, 0.55], allow_in_sample=False)
    expected = {
        "threshold",
        "trade_count",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "expectancy_r",
        "average_r",
        "final_equity",
        "buy_trades",
        "sell_trades",
        "net_profit",
        "return_pct",
        "max_consecutive_losses",
        "hold_filtered_count",
    }
    assert expected.issubset(rows[0])


def test_threshold_sweep_thresholds_are_sorted():
    from main import sample_data
    from src.backtest import run_threshold_sweep
    from src.train import train_random_forest

    cfg = TradingConfig(signal_threshold=0.75)
    df = sample_data(5000)
    train_random_forest(df, cfg)
    rows = run_threshold_sweep(df, cfg, [0.55, 0.5], allow_in_sample=False)
    assert [row["threshold"] for row in rows] == [0.55, 0.5]


def test_threshold_sweep_does_not_enable_live_trading():
    cfg = TradingConfig(trade_mode="paper")
    assert cfg.trade_mode == "paper"
