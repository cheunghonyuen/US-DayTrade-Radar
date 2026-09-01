
import json
from pathlib import Path
import pandas as pd

from original_mvp_engine_v01 import backtest as original_backtest

# Preserved MVP v0.1 config. No tuning.
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

def _normalize_to_ny_local_naive(df):
    """
    Input replay timestamps are UTC.
    Convert them to America/New_York, then strip tz info so the preserved v0.1
    engine sees the same naive local wall-clock format it was written for.
    No trading rule is changed.
    """
    x = df.copy()
    if "timestamp" not in x.columns:
        raise ValueError("timestamp column missing")
    ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{bad} invalid timestamps")
    ny = ts.dt.tz_convert("America/New_York")
    x["timestamp"] = ny.dt.tz_localize(None)
    return x

def _run(path, out_csv):
    raw = pd.read_csv(path)
    fixed = _normalize_to_ny_local_naive(raw)

    # Schema/parity audit before calling original engine.
    t = pd.to_datetime(fixed["timestamp"], errors="coerce")
    hhmm = t.dt.strftime("%H:%M")
    reg_rows = int(((hhmm >= "09:30") & (hhmm <= "15:50")).sum())
    ny_min = str(hhmm.min())
    ny_max = str(hhmm.max())

    trades, metrics = original_backtest(fixed, dict(ORIGINAL_CONFIG))
    trades.to_csv(out_csv, index=False)
    return metrics, {
        "rows": int(len(fixed)),
        "engine_regular_rows": reg_rows,
        "ny_time_min": ny_min,
        "ny_time_max": ny_max,
        "trades": int(len(trades))
    }

def run_original_tz_fixed_replay(discovery_path, oos_path, outdir, progress_cb=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(5, "Discovery UTC → America/New_York → naive local")
    dmetrics, dparity = _run(
        discovery_path, out / "original_v01_tzfixed_discovery_trades.csv"
    )

    if progress_cb:
        progress_cb(52, "True OOS UTC → America/New_York → naive local")
    ometrics, oparity = _run(
        oos_path, out / "original_v01_tzfixed_oos_trades.csv"
    )

    result = {
        "study": "Exact Original MVP v0.1 · Timezone-Fixed Replay",
        "source_engine": "preserved original MVP v0.1 engine.py",
        "rule_changes": "NONE",
        "input_parity_fix": (
            "UTC timestamps converted to America/New_York and made timezone-naive "
            "before entering the preserved original engine."
        ),
        "original_config": ORIGINAL_CONFIG,
        "discovery": dmetrics,
        "true_oos": ometrics,
        "parity": {"discovery": dparity, "true_oos": oparity},
        "decision_rule": (
            "Judge the original model on the current 125-stock pool and current Discovery/True-OOS periods. "
            "If True OOS PF > 1 and AvgR > 0, it has at least positive OOS edge here; "
            "if PF < 1 / AvgR < 0, the preserved original fails this generalization test."
        )
    }
    (out / "original_v01_tzfixed_replay.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if progress_cb:
        progress_cb(100, "Original v0.1 timezone-fixed replay complete")
    return result
