# MT5 Supervised ML Forex Model

Pipeline Python cho MetaTrader 5:

```text
MT5 -> OHLCV/spread -> features -> RandomForest -> BUY/SELL/HOLD probabilities -> threshold -> backtest -> paper
```

Default config:

- Symbol: `EURUSD`
- Timeframe: `H1`
- Horizon: `10` candles
- Pip threshold: `30`
- Bars: `100000`
- Model: `RandomForestClassifier`
- Trade mode: `paper`
- Signal threshold: `0.75`
- Risk: `1%` equity/trade
- SL: ATR-based
- TP: `2R`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Copy env sample if using MT5 login:

```powershell
Copy-Item .env.example .env
```

## Commands

Use sample data first:

```powershell
.\.venv\Scripts\python main.py train --sample
.\.venv\Scripts\python main.py backtest --sample
.\.venv\Scripts\python main.py signal --sample
```

Fetch MT5 data:

```powershell
.\.venv\Scripts\python main.py fetch --symbol EURUSD --timeframe H1 --bars 100000
```

Train/backtest real fetched CSV:

```powershell
.\.venv\Scripts\python main.py train
.\.venv\Scripts\python main.py backtest
.\.venv\Scripts\python main.py signal
```

## Safety

Default `paper` mode does not send orders. Live trading is blocked in code unless explicit environment + command confirmation are added later.
