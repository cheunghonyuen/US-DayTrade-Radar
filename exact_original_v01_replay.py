
import json
from pathlib import Path
import pandas as pd
from original_mvp_engine_v01 import backtest as original_backtest

ORIGINAL_CONFIG = {
  "starting_equity": 100000,
  "risk_per_trade_pct": 0.005,
  "min_gap_pct": 0.03,
  "max_gap_pct": 0.2,
  "min_rvol": 1.2,
  "opening_range_minutes": 20,
  "entry_start": "10:00",
  "entry_end": "11:30",
  "eod_exit": "15:50",
  "target_r": 1.5,
  "stop_range_fraction": 0.5,
  "slippage_per_share": 0.01,
  "commission_per_share": 0.005,
  "max_trades_per_symbol_per_day": 1
}

def _run(path):
    df = pd.read_csv(path)
    trades, metrics = original_backtest(df, dict(ORIGINAL_CONFIG))
    return trades, metrics

def run_exact_original_v01(discovery_path, oos_path, outdir, progress_cb=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(8, "读取现有 Discovery Replay；调用 preserved MVP v0.1 engine")
    dtrades, dmetrics = _run(discovery_path)
    dtrades.to_csv(out / "original_v01_discovery_trades.csv", index=False)

    if progress_cb:
        progress_cb(55, "读取现有 True OOS Replay；同一 original engine / same config")
    otrades, ometrics = _run(oos_path)
    otrades.to_csv(out / "original_v01_oos_trades.csv", index=False)

    result = {
        "study": "Exact Preserved MVP v0.1 Replay",
        "source": "US_DayTrade_MVP_v0.1.zip / src/engine.py + config.json",
        "engine_mode": "PRESERVED_ORIGINAL_ENGINE_NO_LATER_FILTERS",
        "original_config": ORIGINAL_CONFIG,
        "discovery": dmetrics,
        "true_oos": ometrics,
        "comparison_note": (
            "Same current 125 Stocks-in-Play data and same two time blocks. "
            "No VWAP, no DNA, no Filter Ladder, no later gates, no Tournament setup changes."
        ),
        "interpretation": (
            "If True OOS remains strong, later modifications likely damaged the baseline. "
            "If True OOS is weak, the preserved original baseline itself does not generalize "
            "to this stock pool/time period."
        )
    }
    (out / "exact_original_v01_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if progress_cb:
        progress_cb(100, "Exact preserved MVP v0.1 replay complete")
    return result
