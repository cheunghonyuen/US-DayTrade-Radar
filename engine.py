
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class Trade:
    symbol: str
    date: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop: float
    target: float
    shares: int
    pnl: float
    pnl_r: float
    exit_reason: str
    gap_pct: float
    rvol: float

def _to_et_naive(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    required = {"timestamp","symbol","open","high","low","close","volume"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x["timestamp"] = _to_et_naive(x["timestamp"])
    x = x.dropna(subset=["timestamp","open","high","low","close","volume"]).copy()
    x = x.sort_values(["symbol","timestamp"]).drop_duplicates(["symbol","timestamp"]).reset_index(drop=True)

    x["date"] = x["timestamp"].dt.date.astype(str)
    x["time"] = x["timestamp"].dt.strftime("%H:%M")
    x["minute_key"] = x["timestamp"].dt.strftime("%H:%M")

    # Reject invalid bars before any trading logic.
    x = x[
        (x["open"] > 0) & (x["high"] > 0) & (x["low"] > 0) &
        (x["close"] > 0) & (x["volume"] >= 0) &
        (x["high"] >= x["low"])
    ].copy()

    # RVOL must not leak current bar into its own historical baseline.
    x["vol_baseline"] = (
        x.groupby(["symbol","minute_key"])["volume"]
         .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    x["rvol_bar"] = x["volume"] / x["vol_baseline"].replace(0, np.nan)
    return x

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

def backtest(df: pd.DataFrame, cfg: Dict):
    df = prepare(df)
    equity = float(cfg["starting_equity"])
    trades: List[Trade] = []

    for (symbol, date), day in df.groupby(["symbol","date"], sort=True):
        day = day.copy().reset_index(drop=True)
        regular = day[(day["time"] >= "09:30") & (day["time"] < "16:00")].copy()
        if len(regular) < 20:
            continue

        sym_df = df[df["symbol"] == symbol]
        prev_close = _previous_regular_close(sym_df, date)
        if prev_close is None or prev_close <= 0:
            continue

        open_px = float(regular.iloc[0]["open"])
        gap_pct = open_px / prev_close - 1.0
        if not (cfg["min_gap_pct"] <= gap_pct <= cfg["max_gap_pct"]):
            continue

        or_start = pd.Timestamp(f"{date} 09:30")
        or_end = or_start + pd.Timedelta(minutes=int(cfg["opening_range_minutes"]))
        orb = regular[
            (regular["timestamp"] >= or_start) &
            (regular["timestamp"] < or_end)
        ]
        if orb.empty:
            continue

        or_high = float(orb["high"].max())
        or_low = float(orb["low"].min())
        or_range = max(or_high - or_low, 0.01)

        entry_window = regular[
            (regular["time"] >= cfg["entry_start"]) &
            (regular["time"] <= cfg["entry_end"])
        ]
        if entry_window.empty:
            continue

        triggered = False
        for _, row in entry_window.iterrows():
            if triggered:
                break

            if float(row["high"]) <= or_high:
                continue

            rvol = float(row["rvol_bar"]) if pd.notna(row["rvol_bar"]) else np.nan
            if not np.isfinite(rvol) or rvol < float(cfg["min_rvol"]):
                continue

            # Next-bar fill only: signal bar cannot fill itself.
            next_rows = regular[regular["timestamp"] > row["timestamp"]]
            if next_rows.empty:
                continue
            ent_row = next_rows.iloc[0]

            raw_entry = max(float(ent_row["open"]), or_high)
            entry = raw_entry + float(cfg["slippage_per_share"])

            stop = max(
                or_low,
                entry - float(cfg["stop_range_fraction"]) * or_range
            )
            risk_per_share = max(entry - stop, 0.01)

            risk_dollars = equity * float(cfg["risk_per_trade_pct"])
            shares = int(risk_dollars // risk_per_share)
            if shares <= 0:
                continue

            target = entry + float(cfg["target_r"]) * risk_per_share

            post = regular[regular["timestamp"] >= ent_row["timestamp"]].copy()
            exit_px = None
            exit_time = None
            reason = None

            for _, bar in post.iterrows():
                lo = float(bar["low"])
                hi = float(bar["high"])

                # Conservative: if stop and target occur in same bar, stop wins.
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

            trades.append(Trade(
                symbol=symbol,
                date=date,
                entry_time=str(ent_row["timestamp"]),
                exit_time=str(exit_time),
                entry=round(entry, 4),
                exit=round(exit_px, 4),
                stop=round(stop, 4),
                target=round(target, 4),
                shares=shares,
                pnl=round(pnl, 2),
                pnl_r=round(pnl_r, 4),
                exit_reason=reason,
                gap_pct=round(gap_pct, 4),
                rvol=round(rvol, 3),
            ))
            triggered = True

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    return tdf, calc_metrics(tdf, float(cfg["starting_equity"]))

def calc_metrics(tdf: pd.DataFrame, starting_equity: float):
    if tdf.empty:
        return {
            "trades": 0, "win_rate": 0, "profit_factor": 0,
            "net_pnl": 0, "avg_trade": 0, "avg_r": 0,
            "max_drawdown_pct": 0, "stop_rate": 0, "target_rate": 0
        }

    wins = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
    losses = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
    pf = wins / losses if losses > 0 else float("inf")

    eq = starting_equity + tdf["pnl"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak

    return {
        "trades": int(len(tdf)),
        "win_rate": round(float((tdf["pnl"] > 0).mean()), 4),
        "profit_factor": round(float(pf), 4) if np.isfinite(pf) else "inf",
        "net_pnl": round(float(tdf["pnl"].sum()), 2),
        "avg_trade": round(float(tdf["pnl"].mean()), 2),
        "avg_r": round(float(tdf["pnl_r"].mean()), 4),
        "max_drawdown_pct": round(float(dd.min() * 100), 3),
        "stop_rate": round(float((tdf["exit_reason"] == "STOP").mean()), 4),
        "target_rate": round(float((tdf["exit_reason"] == "TARGET").mean()), 4),
    }
