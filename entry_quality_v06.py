import json
from pathlib import Path

import numpy as np
import pandas as pd

RULES = [
    {
        "name": "L0 V0.5 Frozen Baseline",
        "desc": "No new gate",
        "prior3_max": None,
        "or_min": None,
    },
    {
        "name": "L1 Anti-Chase",
        "desc": "prior 3-bar return <= 0.75%",
        "prior3_max": 0.75,
        "or_min": None,
    },
    {
        "name": "L2 OR Quality",
        "desc": "OR range >= 3.50%",
        "prior3_max": None,
        "or_min": 3.50,
    },
    {
        "name": "L3 Anti-Chase + OR",
        "desc": "prior 3-bar return <= 0.75% AND OR range >= 3.50%",
        "prior3_max": 0.75,
        "or_min": 3.50,
    },
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _apply_rule(df, rule):
    m = pd.Series(True, index=df.index)
    if rule["prior3_max"] is not None:
        m &= _num(df["prior_3bar_return_pct"]).le(float(rule["prior3_max"]))
    if rule["or_min"] is not None:
        m &= _num(df["or_range_pct"]).ge(float(rule["or_min"]))
    return df[m].copy()


def _maxdd_from_r(g, starting_equity, risk_pct):
    if g.empty:
        return 0.0, float(starting_equity)
    eq = float(starting_equity)
    peak = eq
    maxdd = 0.0
    for r in _num(g["pnl_r"]).fillna(0.0).tolist():
        eq *= max(0.000001, 1.0 + float(risk_pct) * float(r))
        peak = max(peak, eq)
        dd = (eq / peak - 1.0) * 100.0
        maxdd = min(maxdd, dd)
    return float(maxdd), float(eq)


def _metrics(g, starting_equity, risk_pct):
    if g.empty:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_r": 0.0, "max_drawdown_pct": 0.0,
            "stop_rate": 0.0, "target_rate": 0.0,
            "ending_equity": float(starting_equity),
        }

    r = _num(g["pnl_r"]).fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    maxdd, ending = _maxdd_from_r(g, starting_equity, risk_pct)

    reasons = g["exit_reason"].astype(str)
    return {
        "trades": int(len(g)),
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(pf), 4),
        "avg_r": round(float(r.mean()), 4),
        "max_drawdown_pct": round(float(maxdd), 3),
        "stop_rate": round(float(reasons.eq("STOP").mean()), 4),
        "target_rate": round(float(reasons.eq("TARGET").mean()), 4),
        "ending_equity": round(float(ending), 2),
    }


def _half_metrics(g, starting_equity, risk_pct):
    if g.empty or "date" not in g.columns:
        return 0.0, 0.0
    d = g.copy()
    d["_date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["_date"]).sort_values("_date")
    if d.empty:
        return 0.0, 0.0
    midpoint = d["_date"].min() + (d["_date"].max() - d["_date"].min()) / 2
    a = _metrics(d[d["_date"] <= midpoint], starting_equity, risk_pct)
    b = _metrics(d[d["_date"] > midpoint], starting_equity, risk_pct)
    return a["profit_factor"], b["profit_factor"]


def run_entry_quality_validation(trades_path: str, config_path: str, outdir: str = None, progress_cb=None):
    tp = Path(trades_path)
    if not tp.exists():
        raise FileNotFoundError(
            f"V0.5 DNA trade cache not found: {tp}. Run V0.5 Winner vs Loser DNA once."
        )

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    starting_equity = float(cfg.get("starting_equity", 100000))
    risk_pct = float(cfg.get("risk_per_trade_pct", 0.005))

    if progress_cb:
        progress_cb(5, "读取 V0.5 DNA Trade Cache")

    df = pd.read_csv(tp)
    required = {
        "pnl_r", "exit_reason", "prior_3bar_return_pct",
        "or_range_pct", "date"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DNA cache schema missing columns: {missing}")

    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["_date", "entry_time" if "entry_time" in df.columns else "date"]).reset_index(drop=True)

    baseline_n = max(len(df), 1)
    rows = []
    baseline = None

    for i, rule in enumerate(RULES, 1):
        if progress_cb:
            progress_cb(10 + int((i-1)/len(RULES)*75), f"验证 {i}/{len(RULES)}：{rule['name']}")

        g = _apply_rule(df, rule)
        m = _metrics(g, starting_equity, risk_pct)
        h1_pf, h2_pf = _half_metrics(g, starting_equity, risk_pct)

        row = {
            "name": rule["name"],
            "rule": rule["desc"],
            **m,
            "retention_pct": round(100.0 * len(g) / baseline_n, 1),
            "h1_pf": round(float(h1_pf), 4),
            "h2_pf": round(float(h2_pf), 4),
        }

        if baseline is None:
            baseline = dict(row)
            row.update({
                "delta_pf": 0.0,
                "delta_avg_r": 0.0,
                "delta_stop_rate_pp": 0.0,
                "delta_maxdd_pp": 0.0,
                "verdict": "FROZEN",
            })
        else:
            row.update({
                "delta_pf": round(row["profit_factor"] - baseline["profit_factor"], 4),
                "delta_avg_r": round(row["avg_r"] - baseline["avg_r"], 4),
                "delta_stop_rate_pp": round((row["stop_rate"] - baseline["stop_rate"]) * 100.0, 2),
                "delta_maxdd_pp": round(row["max_drawdown_pct"] - baseline["max_drawdown_pct"], 3),
            })

            # Deliberately conservative: no winner by PF alone.
            improves_edge = (
                row["profit_factor"] >= baseline["profit_factor"] + 0.10
                and row["avg_r"] > baseline["avg_r"]
                and row["stop_rate"] < baseline["stop_rate"]
            )
            enough_sample = row["retention_pct"] >= 60.0 and row["trades"] >= 100
            dd_ok = row["max_drawdown_pct"] >= baseline["max_drawdown_pct"]
            split_ok = (row["h1_pf"] > 1.0) and (row["h2_pf"] > 1.0)

            if improves_edge and enough_sample and dd_ok and split_ok:
                row["verdict"] = "PASS"
            elif improves_edge and enough_sample:
                row["verdict"] = "WATCH"
            else:
                row["verdict"] = "FAIL"

        rows.append(row)

    if progress_cb:
        progress_cb(92, "生成 A/B 增益与半年度稳定性检查")

    best = None
    candidates = [r for r in rows[1:] if r["verdict"] in ("PASS", "WATCH")]
    if candidates:
        best = sorted(
            candidates,
            key=lambda r: (r["verdict"] == "PASS", r["profit_factor"], r["avg_r"], r["retention_pct"]),
            reverse=True
        )[0]

    result = {
        "study": "V0.6 Entry Quality Validation",
        "source": str(tp),
        "frozen_baseline": "RVOL>=3 / Gap>=3% / OR20 / +VWAP",
        "rules_are_predeclared": True,
        "threshold_sweep": False,
        "rows": rows,
        "best_candidate": best,
        "decision_rule": (
            "PASS requires PF +0.10 or more vs baseline, Avg R higher, Stop Rate lower, "
            "Retention >=60%, Trades >=100, MaxDD not worse, and both half-sample PF >1. "
            "WATCH may improve edge but misses robustness conditions."
        ),
        "note": (
            "This is a gate validation on the frozen V0.5 trade cache. MFE/MAE are not used. "
            "MaxDD is recomputed from pnl_r using frozen risk_per_trade_pct, so skipped trades "
            "do not contaminate position sizing."
        ),
    }

    out = Path(outdir) if outdir else tp.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "entry_quality_v06.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "entry_quality_v06.csv", index=False)

    if progress_cb:
        progress_cb(100, "V0.6 Entry Quality Validation 完成")
    return result
