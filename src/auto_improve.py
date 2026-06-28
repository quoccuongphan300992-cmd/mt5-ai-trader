"""Offline auto-improve search over bounded model/label configs."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import SignalFilters
from .config import TradingConfig
from .data import latest_raw_csv, load_csv
from .train import continue_train_sklearn_ensemble, train_model_from_dataframe
from .walk_forward import WalkForwardSettings, build_thresholds, run_walk_forward

DEFAULT_MIN_TRADES = 30
DEFAULT_MIN_PROFIT_FACTOR = 1.20
DEFAULT_MIN_EXPECTANCY = 0.0
DEFAULT_MIN_POSITIVE_FOLD_RATIO = 0.60
DEFAULT_MAX_DRAWDOWN_LIMIT = 0.20

@dataclass(frozen=True)
class AutoImproveCriteria:
    min_trades: int = DEFAULT_MIN_TRADES
    min_profit_factor: float = DEFAULT_MIN_PROFIT_FACTOR
    min_expectancy: float = DEFAULT_MIN_EXPECTANCY
    min_positive_fold_ratio: float = DEFAULT_MIN_POSITIVE_FOLD_RATIO
    max_drawdown_limit: float = DEFAULT_MAX_DRAWDOWN_LIMIT

@dataclass(frozen=True)
class PromotionConfig:
    promotion_mode: str = "candidate-only"
    candidate_model_dir: str = "models/candidates"
    min_pf_improvement: float = 0.0
    min_trade_improvement: int = 0

@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    round: int
    model_type: str
    label_method: str
    label_atr_tp_mult: float
    label_atr_sl_mult: float
    horizon: int
    direction: str
    filter_preset: str = "none"
    atr_min: float | None = None
    atr_max: float | None = None
    spread_max: float | None = None


FILTER_PRESETS = (
    "none",
    "trend_ema200",
    "atr_mid",
    "london_ny",
    "adx_trend",
    "avoid_chop",
    "trend_atr_combo",
    "spread_safe",
)


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_nested(data: dict[str, Any], key: str) -> Any:
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = get_nested(data, key)
        if value is not None:
            return value
    return None

def extract_previous_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "profit_factor": first_present(metadata, ["profit_factor", "validation_metrics.profit_factor", "walk_forward.profit_factor", "auto_improve.profit_factor"]),
        "trades": first_present(metadata, ["trades", "validation_metrics.trades", "walk_forward.trades", "auto_improve.trades"]),
        "max_drawdown": first_present(metadata, ["max_drawdown", "validation_metrics.max_drawdown", "walk_forward.max_drawdown"]),
    }

def write_safe_manifest(payload: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "safe_auto_improve_manifest.json"
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path

def criteria_payload(criteria: AutoImproveCriteria, promotion_config: PromotionConfig) -> dict[str, Any]:
    return {**asdict(criteria), "min_pf_improvement": promotion_config.min_pf_improvement, "min_trade_improvement": promotion_config.min_trade_improvement}

def extract_winning_config(best: dict[str, Any]) -> dict[str, Any]:
    return {k: best.get(k) for k in ["model_type", "label_method", "label_atr_tp_mult", "label_atr_sl_mult", "horizon", "direction", "filter_preset", "threshold"]}

def extract_validation_metrics(best: dict[str, Any]) -> dict[str, Any]:
    return {k: best.get(k) for k in ["trades", "profit_factor", "expectancy", "positive_fold_ratio", "max_drawdown", "candidate_pass", "fail_reasons"]}

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 10.0 if value > 0 else -10.0
    return value

def _load_data(args: Any) -> pd.DataFrame:
    if getattr(args, "sample", False):
        from main import sample_data
        return sample_data()
    csv_path = getattr(args, "csv", None) or latest_raw_csv()
    if not csv_path:
        raise ValueError("No CSV found. Use --csv or --sample.")
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV missing: {csv_path}")
    return load_csv(str(path))

def _base_config(args: Any) -> TradingConfig:
    return TradingConfig(
        symbol=getattr(args, "symbol", "EURUSD"),
        timeframe=getattr(args, "timeframe", "H1"),
        bars=getattr(args, "bars", None) or 100000,
        horizon=getattr(args, "horizon", None) or 10,
        pip_threshold=getattr(args, "pip_threshold", 30.0),
        signal_threshold=getattr(args, "signal_threshold", 0.75),
        risk_per_trade=getattr(args, "risk", 0.01),
        trade_mode=getattr(args, "trade_mode", "paper"),
    )

def _raw_supports_feature(raw_df: pd.DataFrame | None, feature_name: str) -> bool:
    if raw_df is None:
        return True
    raw_columns = set(raw_df.columns)
    engineered = {
        "ema_200": {"close"},
        "trend_stack_bull": {"close"},
        "trend_stack_bear": {"close"},
        "atr_percentile_100": {"high", "low", "close"},
        "adx_14": {"high", "low", "close"},
        "rolling_std_percentile_100": {"close"},
        "is_london_session": {"time"},
        "is_new_york_session": {"time"},
        "spread_to_atr": {"spread", "high", "low", "close"},
    }
    required = engineered.get(feature_name)
    return feature_name in raw_columns or bool(required and required.issubset(raw_columns))

def _filters_for_candidate(args: Any, candidate: CandidateConfig, raw_df: pd.DataFrame | None = None) -> tuple[SignalFilters, dict[str, Any]]:
    preset = (candidate.filter_preset or getattr(args, "filter_preset", "none") or "none").lower()
    warnings: list[str] = []
    applied = False
    min_atr = candidate.atr_min if candidate.atr_min is not None else getattr(args, "min_atr_percentile", None)
    max_atr = candidate.atr_max if candidate.atr_max is not None else getattr(args, "max_atr_percentile", None)
    max_spread = candidate.spread_max if candidate.spread_max is not None else getattr(args, "max_spread_percentile", None)
    max_spread_to_atr = getattr(args, "max_spread_to_atr", None)
    sessions = None
    require_bear = False
    require_bull = False

    def apply_trend() -> None:
        nonlocal applied, require_bear, require_bull
        if _raw_supports_feature(raw_df, "trend_stack_bull") and _raw_supports_feature(raw_df, "trend_stack_bear"):
            if candidate.direction == "BUY":
                require_bull = True
            elif candidate.direction == "SELL":
                require_bear = True
            applied = True
        else:
            warnings.append("missing_ema200_columns")

    if preset == "none":
        pass
    elif preset == "trend_ema200":
        apply_trend()
    elif preset == "atr_mid":
        if _raw_supports_feature(raw_df, "atr_percentile_100"):
            min_atr = max(min_atr or 0.0, 0.20)
            max_atr = min(max_atr if max_atr is not None else 1.0, 0.80)
            applied = True
        else:
            warnings.append("missing_atr_ratio_column")
    elif preset == "london_ny":
        if _raw_supports_feature(raw_df, "is_london_session") and _raw_supports_feature(raw_df, "is_new_york_session"):
            sessions = ("london", "new_york")
            applied = True
        else:
            warnings.append("missing_datetime_for_session_filter")
    elif preset == "adx_trend":
        if _raw_supports_feature(raw_df, "adx_14"):
            apply_trend()
            warnings.append("adx_14_available_but_signal_filter_not_supported")
        else:
            warnings.append("missing_adx_14")
    elif preset == "avoid_chop":
        if _raw_supports_feature(raw_df, "rolling_std_percentile_100"):
            min_atr = max(min_atr or 0.0, 0.20)
            applied = True
        else:
            warnings.append("missing_bb_width_ratio_column")
    elif preset == "trend_atr_combo":
        apply_trend()
        if _raw_supports_feature(raw_df, "atr_percentile_100"):
            min_atr = max(min_atr or 0.0, 0.20)
            max_atr = min(max_atr if max_atr is not None else 1.0, 0.80)
            applied = True
        else:
            warnings.append("missing_atr_ratio_column")
    elif preset == "spread_safe":
        if _raw_supports_feature(raw_df, "spread_to_atr"):
            max_spread_to_atr = min(max_spread_to_atr if max_spread_to_atr is not None else 0.10, 0.10)
            applied = True
        else:
            warnings.append("missing_spread_column")
    else:
        warnings.append(f"unknown_filter_preset:{preset}")

    filters = SignalFilters(
        min_atr_percentile=min_atr,
        max_atr_percentile=max_atr,
        max_spread_percentile=max_spread,
        max_spread_to_atr=max_spread_to_atr,
        allowed_sessions=sessions,
        require_bear_stack_for_sell=require_bear,
        require_bull_stack_for_buy=require_bull,
    )
    return filters, {"filter_preset": preset, "filter_applied": applied, "filter_warning": "|".join(dict.fromkeys(warnings))}

def _candidate_config(base: TradingConfig, candidate: CandidateConfig) -> TradingConfig:
    return replace(base, model_type=candidate.model_type, label_method=candidate.label_method, label_atr_tp_mult=candidate.label_atr_tp_mult, label_atr_sl_mult=candidate.label_atr_sl_mult, horizon=candidate.horizon)

def build_candidate_grid(args: Any) -> list[CandidateConfig]:
    explicit_filter = getattr(args, "filter_preset", None)
    explicit_filters = [explicit_filter] if explicit_filter and explicit_filter != "grid" else None
    priority_filters = explicit_filters or ["none", "trend_ema200", "atr_mid", "london_ny"]
    expansion_filters = explicit_filters or ["adx_trend", "avoid_chop", "trend_atr_combo", "spread_safe"]
    candidates: list[CandidateConfig] = []
    seen: set[tuple[str, float, float, int, str, str]] = set()
    idx = 1

    model_types = ["extra_trees", "random_forest"]
    if getattr(args, "include_heavy_models", False):
        model_types.extend(["lightgbm", "xgboost"])

    def append_grid(tp_values: list[float], sl_values: list[float], horizons: list[int], directions: list[str], presets: list[str]) -> None:
        nonlocal idx
        for model_type in model_types:
            for tp_mult in tp_values:
                for sl_mult in sl_values:
                    for horizon in horizons:
                        for direction in directions:
                            for preset in presets:
                                key = (model_type, float(tp_mult), float(sl_mult), int(horizon), direction, preset)
                                if key in seen:
                                    continue
                                seen.add(key)
                                candidates.append(CandidateConfig(f"auto_{idx:04d}", idx, model_type, "atr_path", float(tp_mult), float(sl_mult), int(horizon), direction, preset))
                                idx += 1

    append_grid([2.0, 2.5], [1.0, 1.2], [8, 12], ["BUY", "SELL"], priority_filters)
    append_grid([1.5, 3.0], [0.8, 1.5], [6, 18, 24], ["BUY", "SELL"], expansion_filters)
    return candidates

def normalize_walk_forward_rows(raw_rows: list[dict[str, Any]] | pd.DataFrame, candidate: CandidateConfig, filter_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records = raw_rows.to_dict(orient="records") if isinstance(raw_rows, pd.DataFrame) else list(raw_rows or [])
    rows: list[dict[str, Any]] = []
    for raw in records:
        folds = _safe_int(raw.get("ok_folds", raw.get("folds")), 0)
        positive_folds = max(_safe_int(raw.get("positive_expectancy_folds"), 0), _safe_int(raw.get("positive_pf_folds"), 0))
        positive_ratio = (positive_folds / folds) if folds else math.nan
        drawdown_pct = abs(_safe_float(raw.get("max_fold_drawdown_pct", raw.get("max_drawdown")), math.nan))
        drawdown = drawdown_pct / 100.0 if not math.isnan(drawdown_pct) and drawdown_pct > 1.0 else drawdown_pct
        row = {**asdict(candidate), **(filter_meta or {}), "threshold": _safe_float(raw.get("threshold"), math.nan), "trades": _safe_int(raw.get("total_trades", raw.get("trades")), 0), "wins": _safe_int(raw.get("wins"), 0), "losses": _safe_int(raw.get("losses"), 0), "win_rate": _safe_float(raw.get("win_rate"), math.nan), "profit_factor": _safe_float(raw.get("overall_profit_factor", raw.get("profit_factor")), math.nan), "expectancy": _safe_float(raw.get("overall_expectancy_r", raw.get("expectancy_r", raw.get("expectancy"))), math.nan), "total_return": _safe_float(raw.get("average_return_pct", raw.get("total_return")), math.nan), "max_drawdown": drawdown, "positive_folds": positive_folds, "total_folds": folds, "positive_fold_ratio": positive_ratio, "error": raw.get("error", "")}
        row["fail_reasons"] = []
        row["candidate_pass"] = False
        row["score"] = compute_score(row)
        rows.append(row)
    return rows

def compute_fail_reasons(row: dict[str, Any], criteria: AutoImproveCriteria) -> list[str]:
    reasons: list[str] = []
    if row.get("error"):
        reasons.append("walk_forward_failed")
    required = ["profit_factor", "expectancy", "positive_fold_ratio", "max_drawdown"]
    if any(math.isnan(_safe_float(row.get(key), math.nan)) for key in required):
        reasons.append("missing_metrics")
    if _safe_int(row.get("trades"), 0) < criteria.min_trades:
        reasons.append("too_few_trades")
    if _safe_float(row.get("profit_factor"), -math.inf) < criteria.min_profit_factor:
        reasons.append("pf_too_low")
    if _safe_float(row.get("expectancy"), -math.inf) <= criteria.min_expectancy:
        reasons.append("expectancy_not_positive")
    if _safe_float(row.get("positive_fold_ratio"), -math.inf) < criteria.min_positive_fold_ratio:
        reasons.append("too_few_positive_folds")
    if abs(_safe_float(row.get("max_drawdown"), math.inf)) > criteria.max_drawdown_limit:
        reasons.append("drawdown_too_high")
    return sorted(set(reasons), key=reasons.index)

def is_candidate_pass(row: dict[str, Any], criteria: AutoImproveCriteria) -> bool:
    return len(compute_fail_reasons(row, criteria)) == 0

def compute_score(row: dict[str, Any]) -> float:
    trades = _safe_int(row.get("trades"), 0)
    pf = _safe_float(row.get("profit_factor"), 0.0)
    pf = 0.0 if math.isnan(pf) else (2.0 if math.isinf(pf) else max(0.0, min((pf - 1.0) / 0.5, 2.0)))
    exp = _safe_float(row.get("expectancy"), 0.0)
    exp = 0.0 if math.isnan(exp) or math.isinf(exp) else max(0.0, min(exp / 0.10, 2.0))
    ratio = _safe_float(row.get("positive_fold_ratio"), 0.0)
    ratio = 0.0 if math.isnan(ratio) or math.isinf(ratio) else max(0.0, min(ratio, 1.0))
    dd = abs(_safe_float(row.get("max_drawdown"), 0.25))
    dd = 0.25 if math.isnan(dd) or math.isinf(dd) else dd
    dd_score = max(0.0, 1.0 - (dd / 0.25))
    trade_score = min(max(trades, 0), 500) / 500.0
    score = 4.0 * ratio + 3.0 * pf + 2.0 * exp + 1.5 * dd_score + 1.5 * trade_score
    if trades < 100:
        score *= 0.25
    elif trades < 250:
        score *= 0.60
    if ratio < 0.5:
        score *= 0.50
    return float(score)

def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (bool(r.get("candidate_pass")), _safe_float(r.get("score"), -math.inf), _safe_float(r.get("profit_factor"), -math.inf), _safe_float(r.get("expectancy"), -math.inf), _safe_int(r.get("trades"), 0), _safe_float(r.get("positive_fold_ratio"), -math.inf), -abs(_safe_float(r.get("max_drawdown"), math.inf))), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked

def build_fail_reason_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {k: 0 for k in ["too_few_trades", "pf_too_low", "expectancy_not_positive", "too_few_positive_folds", "drawdown_too_high", "walk_forward_failed", "missing_metrics"]}
    best_by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        for reason in row.get("fail_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
        cid = str(row.get("candidate_id"))
        if cid not in best_by_candidate or _safe_float(row.get("score"), -math.inf) > _safe_float(best_by_candidate[cid].get("score"), -math.inf):
            best_by_candidate[cid] = row
    return {"total_rows": len(rows), "passing_rows": sum(1 for r in rows if r.get("candidate_pass") is True), "failing_rows": sum(1 for r in rows if r.get("candidate_pass") is not True), "reason_counts": counts, "by_candidate": [{"candidate_id": cid, "best_threshold": row.get("threshold"), "candidate_pass": bool(row.get("candidate_pass")), "fail_reasons": row.get("fail_reasons", [])} for cid, row in best_by_candidate.items()]}

def write_auto_improve_reports(rows: list[dict[str, Any]], best: dict[str, Any] | None, fail_reason_summary: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in rows:
        out = dict(row)
        out["fail_reasons"] = "|".join(out.get("fail_reasons", []))
        csv_rows.append(_json_safe(out))
    columns = ["rank", "round", "candidate_id", "candidate_pass", "score", "fail_reasons", "model_type", "label_method", "label_atr_tp_mult", "label_atr_sl_mult", "horizon", "direction", "filter_preset", "filter_applied", "filter_warning", "threshold", "trades", "wins", "losses", "win_rate", "profit_factor", "expectancy", "total_return", "max_drawdown", "positive_folds", "total_folds", "positive_fold_ratio", "atr_min", "atr_max", "spread_max", "error"]
    pd.DataFrame(csv_rows, columns=columns).to_csv(reports_dir / "auto_improve_candidates.csv", index=False)
    (reports_dir / "auto_improve_candidates.json").write_text(json.dumps(_json_safe(rows), indent=2), encoding="utf-8")
    (reports_dir / "auto_improve_best.json").write_text(json.dumps(_json_safe(best or {"candidate_pass": False, "reason": "no_candidates"}), indent=2), encoding="utf-8")
    (reports_dir / "auto_improve_fail_reasons.json").write_text(json.dumps(_json_safe(fail_reason_summary), indent=2), encoding="utf-8")

def evaluate_candidate(args: Any, candidate: CandidateConfig, criteria: AutoImproveCriteria, raw_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    try:
        df = raw_df if raw_df is not None else _load_data(args)
        cfg = _candidate_config(_base_config(args), candidate)
        thresholds = build_thresholds(None, getattr(args, "min"), getattr(args, "max"), getattr(args, "step"))
        filters, filter_meta = _filters_for_candidate(args, candidate, raw_df=df)
        settings = WalkForwardSettings(thresholds=thresholds, direction=candidate.direction, folds=getattr(args, "folds", 5), initial_train_pct=getattr(args, "initial_train_pct", 0.50), test_pct=getattr(args, "test_pct", 0.10), filters=filters)
        rows = normalize_walk_forward_rows(run_walk_forward(df, cfg, settings, report_prefix=candidate.candidate_id), candidate, filter_meta)
        if not rows:
            raise ValueError("walk-forward produced no threshold rows")
    except Exception as exc:
        rows = [{**asdict(candidate), "filter_applied": False, "filter_warning": str(exc)[:300], "threshold": math.nan, "trades": 0, "wins": 0, "losses": 0, "win_rate": math.nan, "profit_factor": math.nan, "expectancy": math.nan, "total_return": math.nan, "max_drawdown": math.nan, "positive_folds": 0, "total_folds": 0, "positive_fold_ratio": math.nan, "error": str(exc)[:300]}]
    for row in rows:
        row["fail_reasons"] = compute_fail_reasons(row, criteria)
        row["candidate_pass"] = is_candidate_pass(row, criteria)
        row["score"] = compute_score(row)
    return rows


def find_candidate_artifacts(
    *,
    candidate_id: str,
    candidate_model_dir: str | Path,
    candidate_model_path: str | Path | None = None,
    candidate_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve candidate model/metadata paths safely."""
    candidate_dir: Path | None = None
    model_path = Path(candidate_model_path) if candidate_model_path else None
    metadata_path = Path(candidate_metadata_path) if candidate_metadata_path else None
    if model_path is None:
        candidate_dir = Path(candidate_model_dir) / candidate_id
        model_path = candidate_dir / "model.joblib"
    else:
        candidate_dir = model_path.parent
    if metadata_path is None:
        inferred = candidate_dir / "metadata.json" if candidate_dir else None
        metadata_path = inferred if inferred and inferred.exists() else None
    if not model_path.exists():
        raise FileNotFoundError(f"Candidate model missing: {model_path}")
    metadata = read_json_if_exists(metadata_path) if metadata_path else {}
    return {"candidate_id": candidate_id, "model_path": model_path, "metadata_path": metadata_path, "candidate_dir": candidate_dir, "metadata": metadata}


def _candidate_report_row(candidate_id: str, reports_dir: str | Path = "reports") -> dict[str, Any]:
    path = Path(reports_dir) / "auto_improve_candidates.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "candidate_id" not in df.columns:
        return {}
    rows = df[df["candidate_id"].astype(str) == str(candidate_id)]
    if rows.empty:
        return {}
    if "rank" in rows.columns:
        rows = rows.sort_values("rank", ascending=True)
    return rows.iloc[0].dropna().to_dict()


def continue_train_candidate(args: Any) -> dict[str, Any]:
    """CLI entrypoint for continuing an existing auto-improve candidate."""
    artifacts = find_candidate_artifacts(
        candidate_id=str(args.candidate_id),
        candidate_model_dir=getattr(args, "candidate_model_dir", "models/candidates"),
        candidate_model_path=getattr(args, "candidate_model_path", None),
        candidate_metadata_path=getattr(args, "candidate_metadata_path", None),
    )
    report_row = _candidate_report_row(str(args.candidate_id))
    if report_row and isinstance(artifacts.get("metadata"), dict):
        artifacts["metadata"].setdefault("auto_improve_report_row", report_row)
    manifest = continue_train_sklearn_ensemble(
        csv_path=getattr(args, "csv", None),
        sample=bool(getattr(args, "sample", False)),
        candidate_model_path=artifacts["model_path"],
        candidate_metadata_path=artifacts.get("metadata_path"),
        output_dir=getattr(args, "output_dir", "models/candidates"),
        candidate_id=str(args.candidate_id),
        add_estimators=int(getattr(args, "add_estimators", 300)),
        allow_retrain_fallback=bool(getattr(args, "allow_retrain_fallback", False)),
        symbol=getattr(args, "symbol", None),
        timeframe=getattr(args, "timeframe", None),
        bars=getattr(args, "bars", None),
    )
    print(f"[continue-train-candidate] wrote model: {manifest.get('model_path')}")
    print(f"[continue-train-candidate] wrote metadata: {manifest.get('metadata_path')}")
    print(f"[continue-train-candidate] wrote manifest: {manifest.get('manifest_path')}")
    return _json_safe(manifest)

def train_final_winner(args: Any, best: dict[str, Any]) -> dict[str, Any]:
    return train_winning_candidate(args, best, AutoImproveCriteria(), PromotionConfig())

def train_winning_candidate(args: Any, best: dict[str, Any], criteria: AutoImproveCriteria, promotion_config: PromotionConfig) -> dict[str, Any]:
    df = _load_data(args)
    candidate_id = str(best["candidate_id"])
    candidate_dir = Path(promotion_config.candidate_model_dir) / candidate_id
    candidate_model_path = candidate_dir / "model.joblib"
    candidate_metadata_path = candidate_dir / "metadata.json"
    previous_model_hash = sha256_file("models/model.joblib")
    cfg = replace(_base_config(args), model_type=str(best["model_type"]), label_method=str(best["label_method"]), label_atr_tp_mult=float(best["label_atr_tp_mult"]), label_atr_sl_mult=float(best["label_atr_sl_mult"]), horizon=int(best["horizon"]), signal_threshold=float(best["threshold"]), model_path=str(candidate_model_path), metadata_path=str(candidate_metadata_path))
    metadata_extra = {
        "safe_auto_improve": True,
        "trained_from": "auto-improve",
        "promotion_mode": promotion_config.promotion_mode,
        "candidate_id": candidate_id,
        "previous_model_hash": previous_model_hash,
        "production_model_path": "models/model.joblib",
        "production_metadata_path": "models/metadata.json",
        "candidate_model_path": str(candidate_model_path),
        "candidate_metadata_path": str(candidate_metadata_path),
        "criteria": criteria_payload(criteria, promotion_config),
        "winning_config": extract_winning_config(best),
        "validation_metrics": extract_validation_metrics(best),
        "auto_improve_candidate_id": candidate_id,
        "direction": best.get("direction"),
        "threshold": best.get("threshold"),
        "profit_factor": best.get("profit_factor"),
        "expectancy": best.get("expectancy"),
        "trades": best.get("trades"),
        "positive_fold_ratio": best.get("positive_fold_ratio"),
        "max_drawdown": best.get("max_drawdown"),
        "filter_preset": best.get("filter_preset", "none"),
        "filter_applied": best.get("filter_applied"),
        "filter_warning": best.get("filter_warning", ""),
        "candidate_pass": True,
    }
    bundle = train_model_from_dataframe(df, cfg, save_artifacts=True, model_output_path=candidate_model_path, metadata_output_path=candidate_metadata_path, metadata_extra=metadata_extra)
    candidate_model_hash = sha256_file(candidate_model_path)
    if candidate_metadata_path.exists():
        metadata = read_json_if_exists(candidate_metadata_path)
        metadata["candidate_model_hash"] = candidate_model_hash
        metadata["previous_model_hash"] = previous_model_hash
        candidate_metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return {"rows_labeled": bundle.get("metadata", {}).get("rows_labeled"), "candidate_dir": str(candidate_dir), "candidate_model_path": str(candidate_model_path), "candidate_metadata_path": str(candidate_metadata_path), "previous_model_hash": previous_model_hash, "candidate_model_hash": candidate_model_hash, "candidate_model_written": candidate_model_path.exists(), "candidate_metadata_written": candidate_metadata_path.exists()}

def previous_production_payload() -> dict[str, Any]:
    metadata_path = Path("models/metadata.json")
    metadata = read_json_if_exists(metadata_path)
    metrics = extract_previous_metrics(metadata)
    return {"exists": Path("models/model.joblib").exists(), "metadata_exists": metadata_path.exists(), "profit_factor": metrics.get("profit_factor"), "trades": metrics.get("trades"), "max_drawdown": metrics.get("max_drawdown"), "metadata_path": str(metadata_path)}

def should_promote_candidate(best: dict[str, Any], candidate_artifacts: dict[str, Any], criteria: AutoImproveCriteria, promotion_config: PromotionConfig) -> tuple[bool, list[str], dict[str, Any]]:
    blockers: list[str] = []
    prev = previous_production_payload()
    prev_pf = _safe_float(prev.get("profit_factor"), math.nan)
    prev_trades = _safe_int(prev.get("trades"), 0) if prev.get("trades") is not None else None
    cand_pf = _safe_float(best.get("profit_factor"), -math.inf)
    cand_trades = _safe_int(best.get("trades"), 0)
    comparison_available = not math.isnan(prev_pf) or prev_trades is not None
    pf_improvement = None if math.isnan(prev_pf) else cand_pf - prev_pf
    trade_improvement = None if prev_trades is None else cand_trades - prev_trades
    passes_pf_improvement = True if pf_improvement is None else pf_improvement >= promotion_config.min_pf_improvement
    passes_trade_improvement = True if trade_improvement is None else trade_improvement >= promotion_config.min_trade_improvement
    if promotion_config.promotion_mode != "auto-promote": blockers.append("promotion_mode_not_auto_promote")
    if not best.get("candidate_pass"): blockers.append("candidate_not_passed")
    if not Path(str(candidate_artifacts.get("candidate_model_path", ""))).exists(): blockers.append("candidate_model_missing")
    if not Path(str(candidate_artifacts.get("candidate_metadata_path", ""))).exists(): blockers.append("candidate_metadata_missing")
    current_hash = sha256_file(candidate_artifacts.get("candidate_model_path", ""))
    if not candidate_artifacts.get("candidate_model_hash"): blockers.append("candidate_hash_missing")
    if current_hash != candidate_artifacts.get("candidate_model_hash"): blockers.append("candidate_hash_mismatch")
    for reason in compute_fail_reasons(best, criteria): blockers.append(reason)
    if not passes_pf_improvement: blockers.append("pf_improvement_too_low")
    if not passes_trade_improvement: blockers.append("trade_improvement_too_low")
    comparison = {"previous_model_comparison_available": comparison_available, "pf_improvement": pf_improvement, "trade_improvement": trade_improvement, "passes_pf_improvement": passes_pf_improvement, "passes_trade_improvement": passes_trade_improvement}
    return len(blockers) == 0, sorted(set(blockers), key=blockers.index), comparison

def promote_candidate_to_production(candidate_model_path: str | Path, candidate_metadata_path: str | Path) -> dict[str, bool]:
    model_dst = Path("models/model.joblib")
    meta_dst = Path("models/metadata.json")
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_model = model_dst.with_suffix(".joblib.tmp")
    tmp_meta = meta_dst.with_suffix(".json.tmp")
    shutil.copy2(candidate_model_path, tmp_model)
    shutil.copy2(candidate_metadata_path, tmp_meta)
    tmp_model.replace(model_dst)
    tmp_meta.replace(meta_dst)
    return {"production_model_written": True, "production_metadata_written": True}

def run_auto_improve(args: Any) -> dict[str, Any]:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    criteria = AutoImproveCriteria(getattr(args, "min_trades", DEFAULT_MIN_TRADES), getattr(args, "min_profit_factor", DEFAULT_MIN_PROFIT_FACTOR), getattr(args, "min_expectancy", DEFAULT_MIN_EXPECTANCY), getattr(args, "min_positive_fold_ratio", DEFAULT_MIN_POSITIVE_FOLD_RATIO), getattr(args, "max_drawdown_limit", DEFAULT_MAX_DRAWDOWN_LIMIT))
    promotion_config = PromotionConfig(getattr(args, "promotion_mode", "candidate-only"), getattr(args, "candidate_model_dir", "models/candidates"), getattr(args, "min_pf_improvement", 0.0), getattr(args, "min_trade_improvement", 0))
    raw_df = _load_data(args)
    candidates = build_candidate_grid(args)
    if not candidates:
        payload = {"candidate_pass": False, "reason": "no_candidates", "best_candidate": None, "candidate_artifacts": {}, "final_model_trained": False, "promotion_mode": promotion_config.promotion_mode, "promotion_status": "not_attempted", "production_artifacts_written": []}
        write_auto_improve_reports([], payload, build_fail_reason_summary([]), reports_dir)
        write_safe_manifest(payload, reports_dir)
        return payload
    all_rows: list[dict[str, Any]] = []
    reason = "search_budget_exhausted"
    final_model_result = None
    promotion_status = "not_attempted"
    promotion_blockers: list[str] = []
    comparison: dict[str, Any] = {}
    max_rounds = max(0, min(int(getattr(args, "max_rounds", 30)), len(candidates)))
    for idx, candidate in enumerate(candidates[:max_rounds], start=1):
        print(f"[auto-improve] round {idx}/{max_rounds} candidate={candidate.candidate_id} direction={candidate.direction} model={candidate.model_type} label={candidate.label_method} tp={candidate.label_atr_tp_mult} sl={candidate.label_atr_sl_mult} horizon={candidate.horizon} filter={candidate.filter_preset}")
        all_rows.extend(evaluate_candidate(args, candidate, criteria, raw_df=raw_df))
        ranked = rank_candidate_rows(all_rows)
        best = ranked[0] if ranked else None
        if best:
            fail_text = "|".join(best.get("fail_reasons", [])) or "none"
            print(f"[auto-improve] best threshold={best.get('threshold')} trades={best.get('trades')} pf={best.get('profit_factor')} exp={best.get('expectancy')} pass={str(best.get('candidate_pass')).lower()} fail={fail_text}")
        interim_payload = {"candidate_pass": bool(best and best.get("candidate_pass")), "reason": reason, "best_candidate": best, "candidate_artifacts": {}, "final_model_trained": False, "promotion_mode": promotion_config.promotion_mode, "promotion_status": "not_attempted", "production_artifacts_written": []}
        write_auto_improve_reports(ranked, interim_payload, build_fail_reason_summary(all_rows), reports_dir)
        write_safe_manifest(interim_payload, reports_dir)
        passing_rows = [row for row in ranked if row.get("candidate_pass") is True]
        if passing_rows:
            best = passing_rows[0]
            reason = "candidate_passed"
            print(f"[auto-improve] candidate passed: {best.get('candidate_id')} threshold={best.get('threshold')}")
            print("[auto-improve] training isolated candidate model using winning config")
            final_model_result = train_winning_candidate(args, best, criteria, promotion_config)
            print(f"[auto-improve] wrote candidate model: {final_model_result.get('candidate_model_path')}")
            print(f"[auto-improve] wrote candidate metadata: {final_model_result.get('candidate_metadata_path')}")
            should_promote, promotion_blockers, comparison = should_promote_candidate(best, final_model_result, criteria, promotion_config)
            if should_promote:
                promote_candidate_to_production(final_model_result["candidate_model_path"], final_model_result["candidate_metadata_path"])
                promotion_status = "promoted"
                print("[auto-improve] promoted candidate to production models/model.joblib")
            else:
                promotion_status = "blocked"
                print(f"[auto-improve] promotion blocked: {'|'.join(promotion_blockers)}")
                print("[auto-improve] production model artifacts were not changed")
            break
    ranked = rank_candidate_rows(all_rows)
    best = ranked[0] if ranked else None
    if not final_model_result:
        print(f"[auto-improve] no candidate passed within max_rounds={max_rounds}")
        print("[auto-improve] best candidate saved to reports/auto_improve_best.json")
        print("[auto-improve] production model artifacts were not changed")
    payload = {"candidate_pass": bool(best and best.get("candidate_pass")), "reason": reason, "best_candidate": best, "candidate_artifacts": final_model_result or {}, "final_model_trained": bool(final_model_result), "promotion_mode": promotion_config.promotion_mode, "promotion_status": promotion_status, "promotion_blockers": promotion_blockers, "comparison": comparison, "production_artifacts_written": ["models/model.joblib", "models/metadata.json"] if promotion_status == "promoted" else []}
    write_auto_improve_reports(ranked, payload, build_fail_reason_summary(all_rows), reports_dir)
    write_safe_manifest(payload, reports_dir)
    return _json_safe(payload)
