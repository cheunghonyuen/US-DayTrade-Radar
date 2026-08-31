
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare

FROZEN = {
    "min_rvol": 3.0,
    "min_gap_pct": 0.03,
    "opening_range_minutes": 20,
}
GATE = {
    "prior_3bar_return_pct_max": 0.75,
    "or_range_pct_min": 3.50,
}


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _build_session_cache(prepared: pd.DataFrame, progress_cb=None):
    x = prepared.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.dropna(subset=["timestamp"]).copy()
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")

    regular = x[(x["time"] >= "09:30") & (x["time"] < "16:00")].copy()
    regular.sort_values(["symbol", "timestamp"], inplace=True)

    if progress_cb:
        progress_cb(12, "建立 OOS Session Cache")

    session_last = (
        regular.groupby(["symbol", "date"], sort=True)["close"]
        .last()
        .reset_index(name="session_close")
    )
    session_last["prev_close"] = session_last.groupby("symbol")["session_close"].shift(1)
    prev_map = {
        (str(r.symbol), str(r.date)): float(r.prev_close)
        for r in session_last.itertuples(index=False)
        if pd.notna(r.prev_close) and float(r.prev_close) > 0
    }

    groups = {}
    grouped = list(regular.groupby(["symbol", "date"], sort=True))
    total = len(grouped)
    for i, ((sym, dt), g) in enumerate(grouped, 1):
        g = g.copy().reset_index(drop=True)
        typical = (g["high"] + g["low"] + g["close"]) / 3.0
        denom = g["volume"].cumsum().replace(0, np.nan)
        g["vwap_oos"] = (typical * g["volume"]).cumsum() / denom
        groups[(str(sym), str(dt))] = g
        if progress_cb and (i % 1500 == 0 or i == total):
            progress_cb(12 + int(13 * i / max(total, 1)), f"OOS Session Cache {i}/{total}")
    return groups, prev_map


def _calc_metrics(trades: pd.DataFrame, starting_equity: float):
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "max_drawdown_pct": 0.0,
            "stop_rate": 0.0,
            "target_rate": 0.0,
            "net_pnl": 0.0,
        }

    r = _num(trades["pnl_r"]).fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    pnl = _num(trades["pnl"]).fillna(0.0)
    eq = float(starting_equity)
    peak = eq
    maxdd = 0.0
    for v in pnl:
        eq += float(v)
        peak = max(peak, eq)
        if peak > 0:
            maxdd = min(maxdd, (eq / peak - 1.0) * 100.0)

    reasons = trades["exit_reason"].astype(str)
    return {
        "trades": int(len(trades)),
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(pf), 4),
        "avg_r": round(float(r.mean()), 4),
        "max_drawdown_pct": round(float(maxdd), 3),
        "stop_rate": round(float(reasons.eq("STOP").mean()), 4),
        "target_rate": round(float(reasons.eq("TARGET").mean()), 4),
        "net_pnl": round(float(pnl.sum()), 2),
    }


def _run_replay(groups, prev_map, cfg, use_gate: bool, progress_cb=None, start_pct=25, end_pct=70):
    equity = float(cfg["starting_equity"])
    trades = []
    keys = sorted(groups.keys())
    total = len(keys)

    for i, key in enumerate(keys, 1):
        if progress_cb and (i % 500 == 0 or i == total):
            pct = start_pct + int((end_pct - start_pct) * i / max(total, 1))
            progress_cb(pct, f"{'Frozen Gate' if use_gate else 'Frozen Baseline'} Replay {i}/{total}")

        symbol, date = key
        regular = groups[key]
        if len(regular) < 20:
            continue

        prev_close = prev_map.get(key)
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
        or_mid = (or_high + or_low) / 2.0
        or_range_pct = ((or_high - or_low) / or_mid * 100.0) if or_mid > 0 else np.nan
        or_range = max(or_high - or_low, 0.01)

        entry_window = regular[
            (regular["time"] >= cfg["entry_start"]) &
            (regular["time"] <= cfg["entry_end"])
        ]
        if entry_window.empty:
            continue

        # Evaluate candidates in time order. Gate is applied BEFORE entry.
        chosen = None
        chosen_j = None
        for j, row in entry_window.iterrows():
            if float(row["high"]) <= or_high:
                continue
            if float(row["rvol_bar"]) < float(cfg["min_rvol"]):
                continue
            if not pd.notna(row["vwap_oos"]) or float(row["close"]) <= float(row["vwap_oos"]):
                continue

            if use_gate:
                prev3 = regular.loc[max(0, j-3):j-1]
                if len(prev3) < 2 or float(prev3.iloc[0]["close"]) <= 0:
                    continue
                prior3_ret = (
                    float(prev3.iloc[-1]["close"]) / float(prev3.iloc[0]["close"]) - 1.0
                ) * 100.0
                if prior3_ret > float(GATE["prior_3bar_return_pct_max"]):
                    continue
                if not np.isfinite(or_range_pct) or or_range_pct < float(GATE["or_range_pct_min"]):
                    continue

            chosen = row
            chosen_j = j
            break

        if chosen is None:
            continue

        later = regular[regular["timestamp"] > chosen["timestamp"]]
        if later.empty:
            continue
        ent_row = later.iloc[0]

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
        for bar in post.itertuples(index=False):
            lo = float(bar.low)
            hi = float(bar.high)
            if lo <= stop:
                exit_px = stop - float(cfg["slippage_per_share"])
                exit_time = bar.timestamp
                reason = "STOP"
                break
            if hi >= target:
                exit_px = target - float(cfg["slippage_per_share"])
                exit_time = bar.timestamp
                reason = "TARGET"
                break
            if bar.time >= cfg["eod_exit"]:
                exit_px = float(bar.close) - float(cfg["slippage_per_share"])
                exit_time = bar.timestamp
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

        prev3 = regular.loc[max(0, chosen_j-3):chosen_j-1]
        prior3_ret = np.nan
        if len(prev3) >= 2 and float(prev3.iloc[0]["close"]) > 0:
            prior3_ret = (
                float(prev3.iloc[-1]["close"]) / float(prev3.iloc[0]["close"]) - 1.0
            ) * 100.0

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
            "rvol": round(float(chosen["rvol_bar"]), 3),
            "prior_3bar_return_pct": None if not np.isfinite(prior3_ret) else round(prior3_ret, 5),
            "or_range_pct": None if not np.isfinite(or_range_pct) else round(or_range_pct, 5),
        })

    tdf = pd.DataFrame(trades)
    return tdf, _calc_metrics(tdf, float(cfg["starting_equity"]))


def run_true_oos_v08(data_path: str, config_path: str = "config.json", outdir: str = None, progress_cb=None):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"OOS dataset missing: {data_path}")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg.update(FROZEN)

    if progress_cb:
        progress_cb(3, "读取 Untouched OOS Dataset")

    raw = pd.read_csv(data_path)

    if progress_cb:
        progress_cb(8, "Prepare OOS 一次")
    prepared = prepare(raw)

    groups, prev_map = _build_session_cache(prepared, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(25, "Replay Frozen Baseline")
    baseline_trades, baseline = _run_replay(
        groups, prev_map, cfg, use_gate=False,
        progress_cb=progress_cb, start_pct=25, end_pct=48
    )

    if progress_cb:
        progress_cb(50, "Replay Frozen Gate")
    gate_trades, gate = _run_replay(
        groups, prev_map, cfg, use_gate=True,
        progress_cb=progress_cb, start_pct=50, end_pct=78
    )

    if progress_cb:
        progress_cb(82, "计算 OOS Verdict")

    retention = 100.0 * gate["trades"] / max(baseline["trades"], 1)
    delta_pf = gate["profit_factor"] - baseline["profit_factor"]
    delta_avg_r = gate["avg_r"] - baseline["avg_r"]
    delta_stop_pp = (gate["stop_rate"] - baseline["stop_rate"]) * 100.0

    checks = {
        "gate_trades_ge_40": gate["trades"] >= 40,
        "gate_pf_ge_1_30": gate["profit_factor"] >= 1.30,
        "gate_avg_r_gt_0_10": gate["avg_r"] > 0.10,
        "gate_stop_rate_lt_baseline": gate["stop_rate"] < baseline["stop_rate"],
        "gate_maxdd_not_worse": gate["max_drawdown_pct"] >= baseline["max_drawdown_pct"],
        "delta_pf_ge_0_15": delta_pf >= 0.15,
        "retention_ge_15pct": retention >= 15.0,
    }
    passed = sum(bool(v) for v in checks.values())
    total = len(checks)

    if passed == total:
        verdict = "TRUE_OOS_PASS"
    elif passed >= total - 2:
        verdict = "OOS_PROMISING"
    else:
        verdict = "TRUE_OOS_FAIL"

    result = {
        "study": "V0.8 True OOS Validation",
        "oos_dataset": str(data_path),
        "frozen_baseline": "RVOL>=3 / Gap>=3% / OR20 / +VWAP",
        "frozen_gate": "prior 3-bar return <=0.75% AND OR range >=3.50%",
        "threshold_sweep": False,
        "baseline": baseline,
        "gate": gate,
        "retention_pct": round(retention, 1),
        "delta_pf": round(delta_pf, 4),
        "delta_avg_r": round(delta_avg_r, 4),
        "delta_stop_rate_pp": round(delta_stop_pp, 2),
        "checks": checks,
        "checks_passed": int(passed),
        "checks_total": int(total),
        "verdict": verdict,
        "note": (
            "This period is separate from the 2025-08-30..2026-08-30 discovery/robustness year. "
            "No threshold tuning is allowed after viewing this result."
        ),
    }

    out = Path(outdir) if outdir else data_path.parent
    out.mkdir(parents=True, exist_ok=True)
    baseline_trades.to_csv(out / "v08_oos_baseline_trades.csv", index=False)
    gate_trades.to_csv(out / "v08_oos_gate_trades.csv", index=False)
    (out / "v08_true_oos.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if progress_cb:
        progress_cb(100, f"V0.8 完成：{verdict} ({passed}/{total} checks)")
    return result
