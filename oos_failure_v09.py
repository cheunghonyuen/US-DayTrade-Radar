
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "gap_pct",
    "rvol",
    "prior_3bar_return_pct",
    "or_range_pct",
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _median(x):
    x = _num(x).replace([np.inf, -np.inf], np.nan).dropna()
    return None if x.empty else float(x.median())


def _robust_effect(win_s, lose_s, all_s):
    wm = _median(win_s)
    lm = _median(lose_s)
    allv = _num(all_s).replace([np.inf, -np.inf], np.nan).dropna()
    if wm is None or lm is None or len(allv) < 8:
        return wm, lm, None, None
    iqr = float(allv.quantile(.75) - allv.quantile(.25))
    denom = iqr if abs(iqr) > 1e-12 else 1.0
    return wm, lm, wm - lm, (wm - lm) / denom


def _summary(df):
    if df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "stop_rate": 0.0,
            "target_rate": 0.0,
        }
    r = _num(df["pnl_r"]).fillna(0.0)
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    reasons = df["exit_reason"].astype(str)
    return {
        "trades": int(len(df)),
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(pf), 4),
        "avg_r": round(float(r.mean()), 4),
        "stop_rate": round(float(reasons.eq("STOP").mean()), 4),
        "target_rate": round(float(reasons.eq("TARGET").mean()), 4),
    }


def run_oos_failure_autopsy(
    gate_trades_path: str,
    baseline_trades_path: str = None,
    outdir: str = None,
    progress_cb=None,
):
    gp = Path(gate_trades_path)
    if not gp.exists():
        raise FileNotFoundError(
            f"V0.8 OOS gate trades not found: {gp}. Run V0.8 True OOS once; do not re-download data."
        )

    if progress_cb:
        progress_cb(5, "读取 V0.8 OOS Gate Trades")

    gate = pd.read_csv(gp)
    required = {"symbol", "date", "pnl_r", "exit_reason", "gap_pct", "rvol", "prior_3bar_return_pct", "or_range_pct"}
    missing = sorted(required - set(gate.columns))
    if missing:
        raise ValueError(f"OOS gate trade schema missing columns: {missing}")

    gate["date"] = pd.to_datetime(gate["date"], errors="coerce")
    gate = gate.sort_values(["date", "symbol"]).reset_index(drop=True)

    if progress_cb:
        progress_cb(18, "拆分 Winner / Loser / Target / Stop")

    winners = gate[_num(gate["pnl_r"]) > 0].copy()
    losers = gate[_num(gate["pnl_r"]) <= 0].copy()
    targets = gate[gate["exit_reason"].astype(str).eq("TARGET")].copy()
    stops = gate[gate["exit_reason"].astype(str).eq("STOP")].copy()

    feature_rows = []
    for i, f in enumerate(FEATURES, 1):
        wm, lm, delta, effect = _robust_effect(winners[f], losers[f], gate[f])
        feature_rows.append({
            "feature": f,
            "winner_median": wm,
            "loser_median": lm,
            "delta": delta,
            "robust_effect": effect,
        })
        if progress_cb:
            progress_cb(20 + int(i / len(FEATURES) * 20), f"Winner/Loser DNA：{f}")

    feature_rows.sort(
        key=lambda x: abs(x["robust_effect"]) if x["robust_effect"] is not None else -1,
        reverse=True
    )

    if progress_cb:
        progress_cb(45, "检查 Failure Concentration")

    # Concentration diagnostics are descriptive only, not entry thresholds.
    symbol_stats = (
        gate.groupby("symbol")
        .agg(
            trades=("pnl_r", "size"),
            avg_r=("pnl_r", "mean"),
            stop_rate=("exit_reason", lambda s: float((s.astype(str) == "STOP").mean())),
        )
        .reset_index()
        .sort_values(["avg_r", "trades"], ascending=[True, False])
    )

    month = gate.copy()
    month["month"] = month["date"].dt.to_period("M").astype(str)
    month_stats = (
        month.groupby("month")
        .agg(
            trades=("pnl_r", "size"),
            avg_r=("pnl_r", "mean"),
            win_rate=("pnl_r", lambda s: float((_num(s) > 0).mean())),
            stop_rate=("exit_reason", lambda s: float((s.astype(str) == "STOP").mean())),
        )
        .reset_index()
        .sort_values("month")
    )

    # Check whether the old V0.7 discovery feature directions even persist in OOS.
    def _dir(row):
        if row["robust_effect"] is None:
            return "NO_SIGNAL"
        if row["robust_effect"] > 0.10:
            return "HIGHER_IN_WINNERS"
        if row["robust_effect"] < -0.10:
            return "LOWER_IN_WINNERS"
        return "WEAK"

    for row in feature_rows:
        row["direction"] = _dir(row)

    # Predeclared interpretation.
    meaningful = [r for r in feature_rows if r["robust_effect"] is not None and abs(r["robust_effect"]) >= 0.15]
    if not meaningful:
        verdict = "NO_CLEAR_PREENTRY_DNA"
        next_step = (
            "Do not sweep new thresholds. The OOS gate failure is not explained by the existing pre-entry fields; "
            "next research must add genuinely new causal/market-context features or revisit exit mechanics."
        )
    else:
        verdict = "OOS_FAILURE_DNA_FOUND"
        next_step = (
            "Use the strongest OOS-separated feature only as a research hypothesis. "
            "Do not tune a cutoff on this OOS year. Confirm on a third untouched period or paper-forward sample."
        )

    baseline = None
    bp = Path(baseline_trades_path) if baseline_trades_path else None
    if bp and bp.exists():
        try:
            baseline = _summary(pd.read_csv(bp))
        except Exception:
            baseline = None

    result = {
        "study": "V0.9 OOS Failure Autopsy",
        "source_gate_trades": str(gp),
        "gate_summary": _summary(gate),
        "baseline_summary": baseline,
        "winner_count": int(len(winners)),
        "loser_count": int(len(losers)),
        "target_count": int(len(targets)),
        "stop_count": int(len(stops)),
        "features": feature_rows,
        "worst_symbols": symbol_stats.head(12).round(4).to_dict("records"),
        "monthly_failure_map": month_stats.round(4).to_dict("records"),
        "verdict": verdict,
        "next_step": next_step,
        "guardrail": (
            "No threshold sweep. This OOS period has already been viewed. "
            "Any new cutoff fitted here would contaminate OOS."
        ),
    }

    out = Path(outdir) if outdir else gp.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "v09_oos_failure_autopsy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(feature_rows).to_csv(out / "v09_oos_failure_features.csv", index=False)
    symbol_stats.to_csv(out / "v09_oos_failure_symbols.csv", index=False)
    month_stats.to_csv(out / "v09_oos_failure_months.csv", index=False)

    if progress_cb:
        progress_cb(100, f"V0.9 完成：{verdict}")
    return result
