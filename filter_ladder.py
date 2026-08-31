import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare, calc_metrics


BASELINE = {
    "min_rvol": 3.0,
    "min_gap_pct": 0.03,
    "opening_range_minutes": 20,
}

LADDER = [
    ("L0 Frozen Baseline", []),
    ("L1 + VWAP", ["vwap"]),
    ("L2 + Breakout Volume", ["vwap", "breakout_volume"]),
    ("L3 + Close Strength", ["vwap", "breakout_volume", "close_strength"]),
    ("L4 + OR Quality", ["vwap", "breakout_volume", "close_strength", "or_quality"]),
]


def _previous_regular_close(sym_df: pd.DataFrame, date: str):
    current_day = pd.Timestamp(date)
    prior = sym_df[
        (sym_df["timestamp"].dt.normalize() < current_day) &
        (sym_df["time"] >= "09:30") &
        (sym_df["time"] < "16:00")
    ]
    if prior.empty:
        return None
    prior_date = prior["timestamp"].dt.date.iloc[-1]
    prior_session = prior[prior["timestamp"].dt.date == prior_date]
    if prior_session.empty:
        return None
    return float(prior_session.iloc[-1]["close"])


def _cum_vwap(day: pd.DataFrame):
    typical = (day["high"] + day["low"] + day["close"]) / 3.0
    pv = typical * day["volume"]
    denom = day["volume"].cumsum().replace(0, np.nan)
    return pv.cumsum() / denom


def _passes_filters(row, or_high, or_low, or_vol_median, filters):
    if "vwap" in filters:
        if not (float(row["close"]) > float(row["vwap"])):
            return False

    if "breakout_volume" in filters:
        if not np.isfinite(or_vol_median) or or_vol_median <= 0:
            return False
        # Predeclared single rule, no threshold sweep.
        if float(row["volume"]) < 1.5 * float(or_vol_median):
            return False

    if "close_strength" in filters:
        rng = float(row["high"]) - float(row["low"])
        if rng <= 0:
            return False
        close_pos = (float(row["close"]) - float(row["low"])) / rng
        if not (float(row["close"]) > or_high and close_pos >= 0.70):
            return False

    if "or_quality" in filters:
        mid = (or_high + or_low) / 2.0
        if mid <= 0:
            return False
        or_pct = (or_high - or_low) / mid
        # Avoid extremely dead or extremely chaotic opening ranges.
        # Fixed research gate, deliberately not optimized here.
        if not (0.005 <= or_pct <= 0.05):
            return False

    return True


def run_layer(prepared: pd.DataFrame, cfg: dict, filters):
    equity = float(cfg["starting_equity"])
    trades = []

    for (symbol, date), day in prepared.groupby(["symbol", "date"], sort=True):
        day = day.copy().reset_index(drop=True)
        regular = day[(day["time"] >= "09:30") & (day["time"] < "16:00")].copy()
        if len(regular) < 20:
            continue

        regular["vwap"] = _cum_vwap(regular)

        sym_df = prepared[prepared["symbol"] == symbol]
        prev_close = _previous_regular_close(sym_df, date)
        if prev_close is None or prev_close <= 0:
            continue

        open_px = float(regular.iloc[0]["open"])
        gap_pct = open_px / prev_close - 1.0
        if not (float(cfg["min_gap_pct"]) <= gap_pct <= float(cfg["max_gap_pct"])):
            continue

        or_start = pd.Timestamp(f"{date} 09:30")
        or_end = or_start + pd.Timedelta(minutes=int(cfg["opening_range_minutes"]))
        orb = regular[(regular["timestamp"] >= or_start) & (regular["timestamp"] < or_end)]
        if orb.empty:
            continue

        or_high = float(orb["high"].max())
        or_low = float(orb["low"].min())
        or_range = max(or_high - or_low, 0.01)
        or_vol_median = float(orb["volume"].median()) if len(orb) else np.nan

        entry_window = regular[
            (regular["time"] >= cfg["entry_start"]) &
            (regular["time"] <= cfg["entry_end"])
        ]
        if entry_window.empty:
            continue

        for _, row in entry_window.iterrows():
            if float(row["high"]) <= or_high:
                continue

            rvol = float(row["rvol_bar"]) if pd.notna(row["rvol_bar"]) else np.nan
            if not np.isfinite(rvol) or rvol < float(cfg["min_rvol"]):
                continue

            if not _passes_filters(row, or_high, or_low, or_vol_median, filters):
                continue

            next_rows = regular[regular["timestamp"] > row["timestamp"]]
            if next_rows.empty:
                continue
            ent_row = next_rows.iloc[0]

            raw_entry = max(float(ent_row["open"]), or_high)
            entry = raw_entry + float(cfg["slippage_per_share"])

            stop = max(or_low, entry - float(cfg["stop_range_fraction"]) * or_range)
            risk_per_share = max(entry - stop, 0.01)
            risk_dollars = equity * float(cfg["risk_per_trade_pct"])
            shares = int(risk_dollars // risk_per_share)
            if shares <= 0:
                continue

            target = entry + float(cfg["target_r"]) * risk_per_share
            post = regular[regular["timestamp"] >= ent_row["timestamp"]]

            exit_px = None
            exit_time = None
            reason = None
            for _, bar in post.iterrows():
                lo = float(bar["low"])
                hi = float(bar["high"])
                if lo <= stop:
                    exit_px = stop - float(cfg["slippage_per_share"])
                    exit_time = bar["timestamp"]
                    reason = "STOP"
                    break
                if hi >= target:
                    exit_px = target - float(cfg["slippage_per_share"])
                    exit_time = bar["timestamp"]
                    reason = "TARGET"
                    break
                if bar["time"] >= cfg["eod_exit"]:
                    exit_px = float(bar["close"]) - float(cfg["slippage_per_share"])
                    exit_time = bar["timestamp"]
                    reason = "EOD"
                    break

            if exit_px is None:
                last = post.iloc[-1]
                exit_px = float(last["close"]) - float(cfg["slippage_per_share"])
                exit_time = last["timestamp"]
                reason = "LAST_BAR"

            gross = (exit_px - entry) * shares
            fees = float(cfg["commission_per_share"]) * shares * 2
            pnl = gross - fees
            pnl_r = pnl / (risk_per_share * shares)
            equity += pnl

            trades.append({
                "symbol": symbol,
                "date": date,
                "entry_time": str(ent_row["timestamp"]),
                "exit_time": str(exit_time),
                "entry": round(entry, 4),
                "exit": round(exit_px, 4),
                "stop": round(stop, 4),
                "target": round(target, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "pnl_r": round(pnl_r, 4),
                "exit_reason": reason,
                "gap_pct": round(gap_pct, 4),
                "rvol": round(rvol, 3),
            })
            break

    tdf = pd.DataFrame(trades)
    metrics = calc_metrics(tdf, float(cfg["starting_equity"]))
    return tdf, metrics


def run_filter_ladder(data_path: str, config_path: str = "config.json"):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg["min_rvol"] = BASELINE["min_rvol"]
    cfg["min_gap_pct"] = BASELINE["min_gap_pct"]
    cfg["opening_range_minutes"] = BASELINE["opening_range_minutes"]

    raw = pd.read_csv(data_path)
    prepared = prepare(raw)

    results = []
    baseline_trades = None

    for idx, (name, filters) in enumerate(LADDER):
        trades, metrics = run_layer(prepared, cfg, filters)
        if idx == 0:
            baseline_trades = max(int(metrics.get("trades", 0)), 1)

        results.append({
            "step": idx,
            "name": name,
            "filters": filters,
            "trades": int(metrics.get("trades", 0)),
            "retention_pct": round(100.0 * int(metrics.get("trades", 0)) / baseline_trades, 1),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "avg_r": metrics.get("avg_r", 0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
            "net_pnl": metrics.get("net_pnl", 0),
            "stop_rate": metrics.get("stop_rate", 0),
            "target_rate": metrics.get("target_rate", 0),
        })

    return results
