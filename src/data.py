"""Data loading and cleaning helpers."""
from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return clean_ohlcv(df)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = df.copy()
    data = data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in data.columns:
        data["tick_volume"] = 0
    if "spread" not in data.columns:
        data["spread"] = 0
    return data


def latest_raw_csv() -> str | None:
    files = sorted(Path("data/raw").glob("*.csv"))
    return str(files[-1]) if files else None
