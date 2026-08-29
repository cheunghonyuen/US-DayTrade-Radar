
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_SYMBOLS = [
    "NVDA","TSLA","MSFT","AAPL","AMZN","GOOGL","AVGO","AMD","NFLX","ORCL",
    "WMT","JPM","XOM","UNH","LRCX","AMAT","CRM","PLTR","HOOD","TSM"
]

BASE = "https://api.tiingo.com/tiingo/equity/intraday"

def fetch_symbol(symbol, start, end, token, freq):
    url = f"{BASE}/{symbol}/prices"
    params = {
        "startDate": start,
        "endDate": end,
        "resampleFreq": freq,
        "columns": "open,high,low,close,volume",
        "afterHours": "false",
        "forceFill": "false",
        "token": token,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise ValueError(f"{symbol}: response missing date")
    df = df.rename(columns={"date":"timestamp"})
    df["symbol"] = symbol
    keep = ["timestamp","symbol","open","high","low","close","volume"]
    return df[keep]

def preflight(df):
    required = ["timestamp","symbol","open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)

    report = {
        "rows": int(len(x)),
        "symbols": int(x["symbol"].nunique()),
        "start_utc": str(x["timestamp"].min()),
        "end_utc": str(x["timestamp"].max()),
        "bad_timestamp": int(x["timestamp"].isna().sum()),
        "duplicates": int(x.duplicated(["symbol","timestamp"]).sum()),
        "bad_price_rows": int((
            (x["open"] <= 0) | (x["high"] <= 0) |
            (x["low"] <= 0) | (x["close"] <= 0) |
            (x["high"] < x["low"])
        ).sum()),
        "negative_volume_rows": int((x["volume"] < 0).sum()),
    }
    return report

def main():
    p = argparse.ArgumentParser(description="Download Tiingo consolidated US equity intraday bars")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--freq", default="5min", choices=["1min","5min","15min"])
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--out", default="daytrade_data.csv")
    args = p.parse_args()

    token = os.getenv("TIINGO_TOKEN")
    if not token:
        raise RuntimeError(
            "TIINGO_TOKEN is not set. Do not paste the token into code. "
            "Set it as an environment variable / GitHub Codespaces secret."
        )

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    frames = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol} ...", flush=True)
        try:
            df = fetch_symbol(symbol, args.start, args.end, token, args.freq)
            if df.empty:
                print("  no data")
                continue
            frames.append(df)
            print(f"  {len(df):,} bars")
        except Exception as e:
            print(f"  FAILED: {e}")
        time.sleep(0.25)

    if not frames:
        raise RuntimeError("No data downloaded.")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["symbol","timestamp"]).sort_values(["symbol","timestamp"])
    report = preflight(data)

    print("\nPREFLIGHT")
    for k, v in report.items():
        print(f"{k}: {v}")

    if report["bad_timestamp"] or report["duplicates"] or report["bad_price_rows"] or report["negative_volume_rows"]:
        raise RuntimeError("Preflight failed. Fix data before backtest.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
