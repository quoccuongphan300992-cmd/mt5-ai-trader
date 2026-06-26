"""MetaTrader 5 data client."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .config import TIMEFRAME_MAP


def _mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package not installed. Run pip install -r requirements.txt") from exc
    return mt5


def initialize_mt5() -> None:
    load_dotenv()
    mt5 = _mt5()
    path = os.getenv("MT5_PATH", "").strip()

    if path:
        mt5_path = Path(path)
        if mt5_path.exists():
            ok = mt5.initialize(path=str(mt5_path))
        else:
            print(f"[WARN] MT5_PATH does not exist: {path}")
            print("[WARN] Falling back to mt5.initialize() auto-detect.")
            ok = mt5.initialize()
    else:
        ok = mt5.initialize()

    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    _mt5().shutdown()


def fetch_rates(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    mt5 = _mt5()
    tf_name = TIMEFRAME_MAP.get(timeframe.upper())
    if not tf_name:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    mt5_timeframe = getattr(mt5, tf_name)

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Symbol not available in MT5: {symbol}")

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time").reset_index(drop=True)


def save_rates(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    os.makedirs("data/raw", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"data/raw/{symbol}_{timeframe}_{stamp}.csv"
    df.to_csv(path, index=False)
    return path
