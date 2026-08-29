import argparse
import json
from pathlib import Path

import pandas as pd

from engine import backtest


def main():
    parser = argparse.ArgumentParser(description="US DayTrade MVP v0.1 Backtest")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--outdir", default="outputs", help="Directory for results")
    args = parser.parse_args()

    config_path = Path(args.config)
    data_path = Path(args.data)
    outdir = Path(args.outdir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    df = pd.read_csv(data_path)

    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))

    trades, metrics = backtest(df, cfg)

    outdir.mkdir(parents=True, exist_ok=True)
    trades_path = outdir / "trades.csv"
    metrics_path = outdir / "metrics.json"

    trades.to_csv(trades_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved trades: {trades_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
