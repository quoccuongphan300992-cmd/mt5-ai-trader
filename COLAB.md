# Google Colab Workflow

Use Colab only for offline training, backtest, and walk-forward validation.
Keep MT5 terminal on local Windows.

Colab is Linux, so it cannot run the Windows MT5 terminal flow. It should read CSV files, train models, write reports, then you download `models/model.joblib` and `models/metadata.json` back to Windows.

## Safety status

```text
Research/backtest only
No demo
No live
No order execution from Colab
```

## Local Windows: export data

Fetch/export MT5 data locally, then upload CSV to Google Drive:

```text
MyDrive/mt5-ai/data/EURUSD_H1.csv
```

Do not upload broker credentials, `.env`, account IDs, passwords, or keys.

## GitHub setup

Recommended `.gitignore` entries:

```text
.env
*.key
models/
reports/
data/
__pycache__/
.venv/
```

Current project already ignores `.env`, `.venv`, `__pycache__`, model artifacts, reports, and raw/processed data.

## Colab notebook cells

### 1. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone project

Change `USERNAME` and repo name.

```python
%cd /content
!rm -rf mt5-ai-trader
!git clone https://github.com/USERNAME/mt5-ai-trader.git
%cd /content/mt5-ai-trader
!ls
```

### 3. Check machine

```python
!python --version
!nproc
!free -h
!df -h
```

Optional GPU check:

```python
!nvidia-smi
```

For RandomForest/scikit-learn tabular work, CPU is usually enough.

### 4. Install Colab-safe dependencies

Preferred:

```python
!pip install -r requirements-colab.txt
```

Fallback if `requirements-colab.txt` is missing:

```python
!grep -v -E "^(MetaTrader5|pywin32|pypiwin32)" requirements.txt > requirements-colab.txt
!pip install -r requirements-colab.txt
```

`MetaTrader5` is Windows/terminal-specific and should stay local.

### 5. Copy CSV from Drive

```python
!mkdir -p data
!cp "/content/drive/MyDrive/mt5-ai/data/EURUSD_H1.csv" data/EURUSD_H1.csv
!ls -lh data
```

Check CSV exists before running long jobs:

```python
import os

csv_path = "data/EURUSD_H1.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

print("CSV OK:", csv_path, os.path.getsize(csv_path), "bytes")
```

### 6. Train from CSV

```python
!python main.py train --csv data/EURUSD_H1.csv --symbol EURUSD --timeframe H1 --bars 50000
```

Check model artifacts:

```python
!ls -lh models
```

Expected:

```text
model.joblib
metadata.json
```

### 7. Walk-forward validation

Candidate narrow sweep:

```python
!python main.py walk-forward --csv data/EURUSD_H1.csv --symbol EURUSD --timeframe H1 --bars 50000 --direction SELL --min 0.48 --max 0.50 --step 0.01
```

Wider sweep:

```python
!python main.py walk-forward --csv data/EURUSD_H1.csv --symbol EURUSD --timeframe H1 --bars 50000 --direction SELL --min 0.46 --max 0.52 --step 0.01
```

Check reports:

```python
!ls -lh reports
```

### 8. Save outputs to Drive

```python
!mkdir -p "/content/drive/MyDrive/mt5-ai/outputs/models"
!mkdir -p "/content/drive/MyDrive/mt5-ai/outputs/reports"

!cp -r models/* "/content/drive/MyDrive/mt5-ai/outputs/models/" 2>/dev/null || true
!cp -r reports/* "/content/drive/MyDrive/mt5-ai/outputs/reports/" 2>/dev/null || true

!echo "Done. Outputs:"
!find "/content/drive/MyDrive/mt5-ai/outputs" -maxdepth 3 -type f -print
```

### 9. Zip outputs for quick download

```python
!cd /content/mt5-ai-trader && zip -r mt5_ai_outputs.zip models reports
!cp mt5_ai_outputs.zip "/content/drive/MyDrive/mt5-ai/outputs/mt5_ai_outputs.zip"
!ls -lh "/content/drive/MyDrive/mt5-ai/outputs/mt5_ai_outputs.zip"
```

Download this file to Windows:

```text
MyDrive/mt5-ai/outputs/mt5_ai_outputs.zip
```

Then extract and copy into local project:

```text
models/model.joblib
models/metadata.json
```

Download these back to Windows if needed:

```text
models/model.joblib
models/metadata.json
reports/walk_forward_summary.csv
reports/walk_forward_folds.csv
reports/walk_forward_trades.csv
```

## Decision rule

Only consider paper if all pass:

```text
candidate_pass = true
total_trades >= 60
positive_expectancy_folds >= 3
positive_pf_folds >= 3
overall_profit_factor > 1.05
overall_expectancy_r > 0
max_fold_drawdown_pct < 20
```

If `candidate_pass=false`, do not run paper/demo/live. Improve features first:

- regime features
- session/time features
- spread/ATR filters
- then try XGBoost

## MT5 import note

This project uses lazy `MetaTrader5` import in `src/mt5_client.py`, so Colab can run train/backtest/walk-forward without installing `MetaTrader5` as long as you do not run `fetch`.
