
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

def _hhmm(ts):
    return ts.strftime("%H:%M")

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    x["date"] = x["timestamp"].dt.date.astype(str)
    x["time"] = x["timestamp"].dt.strftime("%H:%M")
    return x

def backtest(df: pd.DataFrame, cfg: Dict):
    df = prepare(df)
    equity = float(cfg["starting_equity"])
    trades: List[Trade] = []
    daily_symbol_groups = df.groupby(["symbol","date"], sort=True)

    # Rolling 20-session intraday volume baseline at same minute-of-day, approximated
    df["minute_key"] = df["timestamp"].dt.strftime("%H:%M")
    df["vol_baseline"] = (
        df.groupby(["symbol","minute_key"])["volume"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    df["rvol_bar"] = df["volume"] / df["vol_baseline"].replace(0, np.nan)

    for (symbol, date), day in df.groupby(["symbol","date"], sort=True):
        day = day.copy().reset_index(drop=True)
        reg = day[(day["time"] >= "09:30") & (day["time"] <= cfg["eod_exit"])]
        if len(reg) < 30:
            continue

        # Previous close from prior available row before today's regular session
        all_sym = df[df["symbol"] == symbol]
        first_ts = reg["timestamp"].iloc[0]
        prev = all_sym[all_sym["timestamp"] < first_ts]
        if prev.empty:
            continue
        prev_close = float(prev["close"].iloc[-1])
        open_px = float(reg["open"].iloc[0])
        gap_pct = open_px / prev_close - 1.0
        if not (cfg["min_gap_pct"] <= gap_pct <= cfg["max_gap_pct"]):
            continue

        or_end = (pd.Timestamp(f"{date} 09:30") + pd.Timedelta(minutes=cfg["opening_range_minutes"])).strftime("%H:%M")
        orb = reg[(reg["time"] >= "09:30") & (reg["time"] < or_end)]
        if orb.empty:
            continue
        or_high = float(orb["high"].max())
        or_low = float(orb["low"].min())
        or_range = max(or_high - or_low, 0.01)

        entry_window = reg[(reg["time"] >= cfg["entry_start"]) & (reg["time"] <= cfg["entry_end"])]
        if entry_window.empty:
            continue

        # One trade per symbol/day
        triggered = False
        for idx, row in entry_window.iterrows():
            if triggered:
                break
            if float(row["high"]) <= or_high:
                continue
            rvol = float(row["rvol_bar"]) if pd.notna(row["rvol_bar"]) else np.nan
            if not np.isfinite(rvol) or rvol < cfg["min_rvol"]:
                continue

            # next-bar entry to avoid same-bar lookahead
            next_rows = reg[reg["timestamp"] > row["timestamp"]]
            if next_rows.empty:
                continue
            ent_row = next_rows.iloc[0]
            raw_entry = max(float(ent_row["open"]), or_high)
            entry = raw_entry + cfg["slippage_per_share"]

            stop = max(or_low, entry - cfg["stop_range_fraction"] * or_range)
            risk_per_share = max(entry - stop, 0.01)
            risk_dollars = equity * cfg["risk_per_trade_pct"]
            shares = int(risk_dollars // risk_per_share)
            if shares <= 0:
                continue
            target = entry + cfg["target_r"] * risk_per_share

            # simulate after entry
            post = reg[reg["timestamp"] >= ent_row["timestamp"]].copy()
            exit_px = None
            exit_time = None
            reason = None
            for _, bar in post.iterrows():
                lo, hi = float(bar["low"]), float(bar["high"])
                # conservative if stop & target both touched same bar -> stop first
                if lo <= stop:
                    exit_px = stop - cfg["slippage_per_share"]
                    exit_time = bar["timestamp"]
                    reason = "STOP"
                    break
                if hi >= target:
                    exit_px = target - cfg["slippage_per_share"]
                    exit_time = bar["timestamp"]
                    reason = "TARGET"
                    break
                if bar["time"] >= cfg["eod_exit"]:
                    exit_px = float(bar["close"]) - cfg["slippage_per_share"]
                    exit_time = bar["timestamp"]
                    reason = "EOD"
                    break
            if exit_px is None:
                last = post.iloc[-1]
                exit_px = float(last["close"]) - cfg["slippage_per_share"]
                exit_time = last["timestamp"]
                reason = "LAST_BAR"

            gross = (exit_px - entry) * shares
            fees = cfg["commission_per_share"] * shares * 2
            pnl = gross - fees
            pnl_r = pnl / (risk_per_share * shares) if shares else 0.0
            equity += pnl

            trades.append(Trade(
                symbol=symbol, date=date,
                entry_time=str(ent_row["timestamp"]),
                exit_time=str(exit_time),
                entry=round(entry,4), exit=round(exit_px,4),
                stop=round(stop,4), target=round(target,4),
                shares=shares, pnl=round(pnl,2), pnl_r=round(pnl_r,4),
                exit_reason=reason, gap_pct=round(gap_pct,4),
                rvol=round(rvol,3)
            ))
            triggered = True

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    metrics = calc_metrics(tdf, cfg["starting_equity"])
    return tdf, metrics

def calc_metrics(tdf: pd.DataFrame, starting_equity: float):
    if tdf.empty:
        return {
            "trades": 0, "win_rate": 0, "profit_factor": 0,
            "net_pnl": 0, "avg_trade": 0, "avg_r": 0,
            "max_drawdown_pct": 0
        }
    wins = tdf[tdf["pnl"] > 0]["pnl"].sum()
    losses = -tdf[tdf["pnl"] < 0]["pnl"].sum()
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
        "max_drawdown_pct": round(float(dd.min()*100), 3),
        "stop_rate": round(float((tdf["exit_reason"]=="STOP").mean()), 4),
        "target_rate": round(float((tdf["exit_reason"]=="TARGET").mean()), 4),
    }
