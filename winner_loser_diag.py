import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare, calc_metrics

FROZEN = {
    "min_rvol": 3.0,
    "min_gap_pct": 0.03,
    "opening_range_minutes": 20,
}

PRE_ENTRY_FEATURES = [
    "gap_pct",
    "rvol",
    "vwap_dist_pct",
    "or_range_pct",
    "breakout_minute",
    "breakout_extension_pct",
    "prior_3bar_return_pct",
    "prior_3bar_volume_ratio",
]


def _safe_median(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.median()) if len(s) else None


def _feature_deltas(enriched: pd.DataFrame):
    winners = enriched[enriched["pnl_r"] > 0]
    losers = enriched[enriched["pnl_r"] <= 0]
    out = []
    for f in PRE_ENTRY_FEATURES:
        allv = pd.to_numeric(enriched[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        wm = _safe_median(winners[f])
        lm = _safe_median(losers[f])
        if wm is None or lm is None or len(allv) < 10:
            effect = None
        else:
            iqr = float(allv.quantile(.75) - allv.quantile(.25))
            effect = (wm - lm) / (iqr if abs(iqr) > 1e-12 else 1.0)
        out.append({
            "feature": f,
            "winner_median": wm,
            "loser_median": lm,
            "delta": None if wm is None or lm is None else wm - lm,
            "robust_effect": effect,
        })
    out.sort(key=lambda r: abs(r["robust_effect"]) if r["robust_effect"] is not None else -1, reverse=True)
    return out


def _build_session_cache(prepared: pd.DataFrame, progress_cb=None):
    x = prepared.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.dropna(subset=["timestamp"]).copy()
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")

    regular = x[(x["time"] >= "09:30") & (x["time"] < "16:00")].copy()
    regular.sort_values(["symbol", "timestamp"], inplace=True)

    if progress_cb:
        progress_cb(20, "建立 Session Cache / 前收盘映射")

    # Previous regular-session close per symbol/day, computed ONCE.
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
        if progress_cb and (i % 1500 == 0 or i == total):
            progress_cb(20 + int(10 * i / max(total, 1)), f"Session Cache {i}/{total}")

        g = g.copy().reset_index(drop=True)
        typical = (g["high"] + g["low"] + g["close"]) / 3.0
        denom = g["volume"].cumsum().replace(0, np.nan)
        g["vwap_diag"] = (typical * g["volume"]).cumsum() / denom
        groups[(str(sym), str(dt))] = g

    return groups, prev_map


def _fast_frozen_vwap_replay(prepared: pd.DataFrame, cfg: dict, groups, prev_map, progress_cb=None):
    equity = float(cfg["starting_equity"])
    trades = []
    keys = sorted(groups.keys())
    total = len(keys)

    for i, key in enumerate(keys, 1):
        if progress_cb and (i % 500 == 0 or i == total):
            progress_cb(30 + int(25 * i / max(total, 1)), f"冻结基准 Replay {i}/{total}")

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
        or_range = max(or_high - or_low, 0.01)

        entry_window = regular[
            (regular["time"] >= cfg["entry_start"]) &
            (regular["time"] <= cfg["entry_end"])
        ]
        if entry_window.empty:
            continue

        # Same frozen rule as old run_layer: breakout + RVOL>=3 + close > VWAP.
        candidates = entry_window[
            (pd.to_numeric(entry_window["high"], errors="coerce") > or_high) &
            (pd.to_numeric(entry_window["rvol_bar"], errors="coerce") >= float(cfg["min_rvol"])) &
            (pd.to_numeric(entry_window["close"], errors="coerce") >
             pd.to_numeric(entry_window["vwap_diag"], errors="coerce"))
        ]
        if candidates.empty:
            continue

        row = candidates.iloc[0]
        later = regular[regular["timestamp"] > row["timestamp"]]
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

        # Small per-trade loop only over bars AFTER entry.
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
            "rvol": round(float(row["rvol_bar"]), 3),
        })

    tdf = pd.DataFrame(trades)
    metrics = calc_metrics(tdf, float(cfg["starting_equity"]))
    return tdf, metrics


def _build_trade_features(groups, trades: pd.DataFrame, cfg: dict, progress_cb=None):
    if trades.empty:
        return trades.copy()

    rows = []
    total = len(trades)

    for i, tr in trades.reset_index(drop=True).iterrows():
        if progress_cb and (i % 10 == 0 or i + 1 == total):
            progress_cb(55 + int(35 * (i + 1) / max(total, 1)), f"提取交易 DNA {i+1}/{total}")

        sym = str(tr["symbol"])
        dt = str(tr["date"])
        reg = groups.get((sym, dt))
        if reg is None or reg.empty:
            continue

        entry_ts = pd.to_datetime(tr["entry_time"], errors="coerce")
        if pd.isna(entry_ts):
            continue

        arr_ts = reg["timestamp"].values
        j = int(np.searchsorted(arr_ts, np.datetime64(entry_ts), side="left"))
        if j >= len(reg):
            continue
        erow = reg.iloc[j]

        or_start = pd.Timestamp(f"{dt} 09:30")
        or_end = or_start + pd.Timedelta(minutes=int(cfg["opening_range_minutes"]))
        orb = reg[(reg["timestamp"] >= or_start) & (reg["timestamp"] < or_end)]
        if orb.empty:
            continue

        or_high = float(orb["high"].max())
        or_low = float(orb["low"].min())
        or_mid = (or_high + or_low) / 2.0
        or_range_pct = (or_high - or_low) / or_mid * 100.0 if or_mid > 0 else np.nan
        or_vol_median = float(orb["volume"].median()) if len(orb) else np.nan

        entry_px = float(tr["entry"])
        vwap = float(erow["vwap_diag"]) if pd.notna(erow["vwap_diag"]) else np.nan
        vwap_dist_pct = (entry_px / vwap - 1.0) * 100.0 if np.isfinite(vwap) and vwap > 0 else np.nan
        breakout_minute = (entry_ts - pd.Timestamp(f"{dt} 09:30")).total_seconds() / 60.0
        breakout_extension_pct = (entry_px / or_high - 1.0) * 100.0 if or_high > 0 else np.nan

        prev3 = reg.iloc[max(0, j-3):j]
        prior_3bar_return_pct = np.nan
        if len(prev3) >= 2 and float(prev3.iloc[0]["close"]) > 0:
            prior_3bar_return_pct = (float(prev3.iloc[-1]["close"]) / float(prev3.iloc[0]["close"]) - 1.0) * 100.0
        prior_3bar_volume_ratio = (
            float(prev3["volume"].mean()) / or_vol_median
            if len(prev3) and np.isfinite(or_vol_median) and or_vol_median > 0 else np.nan
        )

        risk_per_share = max(float(tr["entry"]) - float(tr["stop"]), 0.01)
        exit_ts = pd.to_datetime(tr["exit_time"], errors="coerce")
        post = reg.iloc[j:]
        if pd.notna(exit_ts):
            post = post[post["timestamp"] <= exit_ts]

        mfe_r = (float(post["high"].max()) - entry_px) / risk_per_share if len(post) else np.nan
        mae_r = (float(post["low"].min()) - entry_px) / risk_per_share if len(post) else np.nan

        out = tr.to_dict()
        out.update({
            "vwap_dist_pct": round(vwap_dist_pct, 5) if np.isfinite(vwap_dist_pct) else None,
            "or_range_pct": round(or_range_pct, 5) if np.isfinite(or_range_pct) else None,
            "breakout_minute": round(breakout_minute, 2),
            "breakout_extension_pct": round(breakout_extension_pct, 5) if np.isfinite(breakout_extension_pct) else None,
            "prior_3bar_return_pct": round(prior_3bar_return_pct, 5) if np.isfinite(prior_3bar_return_pct) else None,
            "prior_3bar_volume_ratio": round(prior_3bar_volume_ratio, 5) if np.isfinite(prior_3bar_volume_ratio) else None,
            "mfe_r_label": round(mfe_r, 4) if np.isfinite(mfe_r) else None,
            "mae_r_label": round(mae_r, 4) if np.isfinite(mae_r) else None,
            "label": "WINNER" if float(tr["pnl_r"]) > 0 else "LOSER",
        })
        rows.append(out)

    return pd.DataFrame(rows)


def run_winner_loser_diagnostic(data_path: str, config_path: str = "config.json", progress_cb=None):
    if progress_cb:
        progress_cb(3, "读取 Persistent Replay Dataset")

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg.update(FROZEN)

    raw = pd.read_csv(data_path)

    if progress_cb:
        progress_cb(10, "Prepare 一次")
    prepared = prepare(raw)

    groups, prev_map = _build_session_cache(prepared, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(30, "Fast Replay：冻结基准 + VWAP")

    trades, metrics = _fast_frozen_vwap_replay(
        prepared, cfg, groups, prev_map, progress_cb=progress_cb
    )

    if progress_cb:
        progress_cb(55, f"Replay 完成：{len(trades)} trades，开始 DNA")

    enriched = _build_trade_features(groups, trades, cfg, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(92, "汇总 Winner / Loser 特征差异")

    winners = enriched[enriched["pnl_r"] > 0]
    losers = enriched[enriched["pnl_r"] <= 0]
    targets = enriched[enriched["exit_reason"] == "TARGET"]
    stops = enriched[enriched["exit_reason"] == "STOP"]

    features = _feature_deltas(enriched)

    result = {
        "frozen_rule": "RVOL>=3.0 / Gap>=3% / OR20 / VWAP long",
        "trades": int(len(enriched)),
        "win_rate": metrics.get("win_rate", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "avg_r": metrics.get("avg_r", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "winner_count": int(len(winners)),
        "loser_count": int(len(losers)),
        "target_count": int(len(targets)),
        "stop_count": int(len(stops)),
        "winner_mfe_r_median": _safe_median(winners["mfe_r_label"]) if len(winners) else None,
        "loser_mfe_r_median": _safe_median(losers["mfe_r_label"]) if len(losers) else None,
        "winner_mae_r_median": _safe_median(winners["mae_r_label"]) if len(winners) else None,
        "loser_mae_r_median": _safe_median(losers["mae_r_label"]) if len(losers) else None,
        "features": features,
        "note": "MFE/MAE are future labels for diagnosis only. They are not allowed as entry filters.",
    }

    outdir = Path(data_path).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    enriched.to_csv(outdir / "winner_loser_trades_v05.csv", index=False)
    (outdir / "winner_loser_diag_v05.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    if progress_cb:
        progress_cb(100, "Winner vs Loser DNA 完成")
    return result
