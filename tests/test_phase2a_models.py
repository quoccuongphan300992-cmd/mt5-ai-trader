import numpy as np
import pandas as pd
import joblib

from src.config import TradingConfig, LABEL_BUY, LABEL_HOLD, LABEL_SELL
from src.signals import OPTIONAL_SIGNAL_COLUMNS, generate_signals
from src.auto_improve import build_candidate_grid, compute_score
from src.backtest import SignalFilters, simulate_signals
from src.train import EncodedLabelClassifier
from src.walk_forward import _probability_columns


class DummyProbabilityModel:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        class_probs = {
            LABEL_BUY: 0.7,
            LABEL_HOLD: 0.2,
            LABEL_SELL: 0.1,
        }
        return np.array([[class_probs[label] for label in self.classes_] for _ in range(len(X))])


def sample_raw_df(rows=260):
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


def test_generate_signals_uses_class_names_for_probability_columns(tmp_path):
    feature_names = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for classes in ([LABEL_BUY, LABEL_HOLD, LABEL_SELL], [LABEL_HOLD, LABEL_SELL, LABEL_BUY]):
        model_path = tmp_path / f"model_{'_'.join(classes)}.joblib"
        joblib.dump({"model": DummyProbabilityModel(classes), "features": feature_names}, model_path)
        cfg = TradingConfig(model_path=str(model_path), signal_threshold=0.6)

        signals = generate_signals(sample_raw_df(), cfg)

        assert float(signals["buy_prob"].iloc[-1]) == 0.7
        assert float(signals["sell_prob"].iloc[-1]) == 0.1
        assert float(signals["hold_prob"].iloc[-1]) == 0.2
        assert signals["signal"].iloc[-1] == LABEL_BUY


def test_auto_improve_grid_excludes_heavy_models_by_default():
    class Args:
        filter_preset = "grid"
        include_heavy_models = False

    model_types = {candidate.model_type for candidate in build_candidate_grid(Args())}

    assert model_types == {"extra_trees", "random_forest"}
    assert "lightgbm" not in model_types
    assert "xgboost" not in model_types


def test_auto_improve_grid_includes_heavy_models_when_flag_enabled():
    class Args:
        filter_preset = "grid"
        include_heavy_models = True

    model_types = {candidate.model_type for candidate in build_candidate_grid(Args())}

    assert {"extra_trees", "random_forest", "lightgbm", "xgboost"}.issubset(model_types)


def test_encoded_label_classifier_exposes_original_string_classes():
    class TinyEstimator:
        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.array(sorted(set(y)))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.2, 0.3, 0.5], (len(X), 1))

    model = EncodedLabelClassifier(TinyEstimator())
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([LABEL_BUY, LABEL_HOLD, LABEL_SELL])

    model.fit(X, y)

    assert set(model.classes_) == {LABEL_BUY, LABEL_HOLD, LABEL_SELL}
    assert not set(model.classes_).issubset({0, 1, 2})
    assert model.predict_proba(X).shape == (3, 3)


def test_auto_improve_targeted_grid_focuses_near_edge_presets():
    class Args:
        filter_preset = "grid"
        grid_mode = "targeted"
        include_heavy_models = False

    candidates = build_candidate_grid(Args())
    combos = {(candidate.direction, candidate.filter_preset) for candidate in candidates}

    assert combos == {("SELL", "trend_ema200"), ("SELL", "london_ny"), ("BUY", "atr_mid")}
    assert {candidate.horizon for candidate in candidates} == {8, 12, 18, 24}
    assert {candidate.label_atr_tp_mult for candidate in candidates} == {1.2, 1.5, 2.0}
    assert {candidate.label_atr_sl_mult for candidate in candidates} == {0.8, 1.0}


def test_compute_score_penalizes_losing_high_trade_candidates():
    losing_high_trade = {
        "trades": 500,
        "profit_factor": 0.99,
        "expectancy": -0.01,
        "positive_fold_ratio": 1.0,
        "max_drawdown": 0.02,
    }
    profitable_low_trade = {
        "trades": 50,
        "profit_factor": 1.20,
        "expectancy": 0.05,
        "positive_fold_ratio": 0.6,
        "max_drawdown": 0.10,
    }

    assert compute_score(losing_high_trade) < compute_score(profitable_low_trade)


def test_generate_signals_propagates_regime_filter_columns(tmp_path):
    feature_names = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": DummyProbabilityModel([LABEL_BUY, LABEL_HOLD, LABEL_SELL]), "features": feature_names}, model_path)
    cfg = TradingConfig(model_path=str(model_path), signal_threshold=0.6)

    signals = generate_signals(sample_raw_df(), cfg)

    for column in [
        "price_above_ema200",
        "price_below_ema200",
        "ema_200_slope_20",
        "adx_14",
        "realized_vol_percentile_100",
    ]:
        assert column in signals.columns


def test_walk_forward_probability_frame_propagates_regime_filter_columns():
    df = sample_raw_df()
    for column in OPTIONAL_SIGNAL_COLUMNS:
        df[column] = 1.0
    bundle = {"model": DummyProbabilityModel([LABEL_BUY, LABEL_HOLD, LABEL_SELL])}

    signals = _probability_columns(bundle, df, ["close"], threshold=0.6)

    for column in [
        "price_above_ema200",
        "price_below_ema200",
        "ema_200_slope_20",
        "adx_14",
        "realized_vol_percentile_100",
    ]:
        assert column in signals.columns


def test_missing_filter_columns_are_reported_not_silent():
    rows = 20
    time = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = pd.Series([1.10 - i * 0.0001 for i in range(rows)])
    signals = pd.DataFrame({
        "time": time,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.0005,
        "low": close - 0.0005,
        "close": close,
        "spread": 10,
        "atr_14": 0.001,
        "buy_prob": 0.1,
        "sell_prob": 0.8,
        "hold_prob": 0.1,
        "signal": LABEL_SELL,
        "confidence": 0.8,
    })
    filters = SignalFilters(require_price_below_ema200_for_sell=True)

    summary, _, _ = simulate_signals(signals, TradingConfig(), direction="SELL", probability_threshold=0.5, filters=filters, write_reports=False)

    assert summary["filter_warning"] == "missing_filter_columns"
    assert "price_below_ema200" in summary["missing_filter_columns"]
