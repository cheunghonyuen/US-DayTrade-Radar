import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare
from filter_ladder import run_layer

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


def _safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else None


def _build_trade_features(prepared: pd.DataFrame, trades: pd.DataFrame, cfg: dict, progress_cb=None):
    if trades.empty:
        return trades.copy()

    x = prepared.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")

    groups = {(sym, dt): g.sort_values("timestamp").reset_index(drop=True)
              for (sym, dt), g in x.groupby(["symbol", "date"], sort=False)}

    rows = []
    total = len(trades)
    for i, tr in trades.reset_index(drop=True).iterrows():
        if progress_cb and (i % 10 == 0 or i + 1 == total):
            progress_cb(35 + int(50 * (i + 1) / max(total, 1)), f"提取交易 DNA {i+1}/{total}")

        sym = str(tr["symbol"])
        dt = str(tr["date"])
        day = groups.get((sym, dt))
        if day is None or day.empty:
            continue

        reg = day[(day["time"] >= "09:30") & (day["time"] < "16:00")].copy().reset_index(drop=True)
        if reg.empty:
            continue

        typical = (reg["high"] + reg["low"] + reg["close"]) / 3.0
        denom = reg["volume"].cumsum().replace(0, np.nan)
        reg["vwap_diag"] = (typical * reg["volume"]).cumsum() / denom

        entry_ts = pd.to_datetime(tr["entry_time"], errors="coerce")
        if pd.isna(entry_ts):
            continue
        hit = reg.index[reg["timestamp"] == entry_ts]
        if len(hit) == 0:
            hit = reg.index[reg["timestamp"] >= entry_ts]
        if len(hit) == 0:
            continue
        j = int(hit[0])
        erow = reg.loc[j]

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
        prior_3bar_volume_ratio = (float(prev3["volume"].mean()) / or_vol_median
                                   if len(prev3) and np.isfinite(or_vol_median) and or_vol_median > 0 else np.nan)

        risk_per_share = max(float(tr["entry"]) - float(tr["stop"]), 0.01)
        exit_ts = pd.to_datetime(tr["exit_time"], errors="coerce")
        post = reg[reg["timestamp"] >= entry_ts]
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


def run_winner_loser_diagnostic(data_path: str, config_path: str = "config.json", progress_cb=None):
    if progress_cb:
        progress_cb(5, "读取固定 Replay Dataset")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg.update(FROZEN)

    raw = pd.read_csv(data_path)
    if progress_cb:
        progress_cb(15, "Prepare 一次并复用")
    prepared = prepare(raw)

    if progress_cb:
        progress_cb(25, "重放冻结基准 + VWAP")
    trades, metrics = run_layer(prepared, cfg, ["vwap"])

    if progress_cb:
        progress_cb(35, f"L1 trades={len(trades)}，提取入场前 DNA")
    enriched = _build_trade_features(prepared, trades, cfg, progress_cb=progress_cb)

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
    (outdir / "winner_loser_diag_v05.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if progress_cb:
        progress_cb(100, "Winner vs Loser DNA 完成")
    return result
