"""Trading execution gates. Paper mode default."""
import os

from .config import TradingConfig


def assert_safe_mode(cfg: TradingConfig, live_confirm: bool = False) -> None:
    mode = cfg.trade_mode.lower()
    if mode == "paper":
        return
    if mode == "demo":
        return
    if mode == "live":
        allowed = os.getenv("ALLOW_LIVE_TRADING", "false").lower() == "true"
        if not (allowed and live_confirm):
            raise RuntimeError("Live trading blocked. Need ALLOW_LIVE_TRADING=true and --i-understand-live-risk")
        return
    raise ValueError(f"Unknown trade mode: {cfg.trade_mode}")


def paper_trade(signal_row: dict) -> dict:
    return {"mode": "paper", "order_sent": False, "signal": signal_row}
