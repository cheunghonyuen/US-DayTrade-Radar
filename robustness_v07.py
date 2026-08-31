
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

GATE = {
    "prior_3bar_return_pct_max": 0.75,
    "or_range_pct_min": 3.50,
}
RNG_SEED = 20260831
BOOTSTRAP_N = 3000


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _stable_bucket(symbol, n=4):
    h = hashlib.sha256(str(symbol).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(n)


def _apply_gate(df):
    return df[
        _num(df["prior_3bar_return_pct"]).le(GATE["prior_3bar_return_pct_max"]) &
        _num(df["or_range_pct"]).ge(GATE["or_range_pct_min"])
    ].copy()


def _metrics(g):
    if g.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "stop_rate": 0.0,
            "target_rate": 0.0,
        }

    r = _num(g["pnl_r"]).fillna(0.0)
    wins = r[r > 0]
    losses = r[r < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    reasons = g["exit_reason"].astype(str)

    return {
        "trades": int(len(g)),
        "win_rate": round(float((r > 0).mean()), 4),
        "profit_factor": round(float(pf), 4),
        "avg_r": round(float(r.mean()), 4),
        "stop_rate": round(float(reasons.eq("STOP").mean()), 4),
        "target_rate": round(float(reasons.eq("TARGET").mean()), 4),
    }


def _compare(base, gate):
    bm = _metrics(base)
    gm = _metrics(gate)
    retention = 100.0 * gm["trades"] / max(bm["trades"], 1)
    return {
        "baseline": bm,
        "gate": gm,
        "retention_pct": round(retention, 1),
        "delta_pf": round(gm["profit_factor"] - bm["profit_factor"], 4),
        "delta_avg_r": round(gm["avg_r"] - bm["avg_r"], 4),
        "delta_stop_rate_pp": round((gm["stop_rate"] - bm["stop_rate"]) * 100.0, 2),
        "delta_win_rate_pp": round((gm["win_rate"] - bm["win_rate"]) * 100.0, 2),
    }


def _bootstrap_gate(gate_df, n_boot=BOOTSTRAP_N):
    if gate_df.empty:
        return {
            "n": 0,
            "avg_r_ci95": [0.0, 0.0],
            "pf_ci95": [0.0, 0.0],
            "prob_avg_r_gt0": 0.0,
            "prob_pf_gt1": 0.0,
        }

    r = _num(gate_df["pnl_r"]).fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    rng = np.random.default_rng(RNG_SEED)
    avg_vals = np.empty(n_boot, dtype=float)
    pf_vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        s = r[rng.integers(0, n, size=n)]
        avg_vals[i] = float(np.mean(s))
        gp = float(s[s > 0].sum())
        gl = float(-s[s < 0].sum())
        pf_vals[i] = gp / gl if gl > 0 else (20.0 if gp > 0 else 0.0)

    return {
        "n": int(n),
        "avg_r_ci95": [round(float(np.quantile(avg_vals, 0.025)), 4),
                       round(float(np.quantile(avg_vals, 0.975)), 4)],
        "pf_ci95": [round(float(np.quantile(pf_vals, 0.025)), 4),
                    round(float(np.quantile(pf_vals, 0.975)), 4)],
        "prob_avg_r_gt0": round(float(np.mean(avg_vals > 0)), 4),
        "prob_pf_gt1": round(float(np.mean(pf_vals > 1)), 4),
    }


def run_robustness_v07(trades_path: str, outdir: str = None, progress_cb=None):
    tp = Path(trades_path)
    if not tp.exists():
        raise FileNotFoundError(
            f"V0.5 DNA trade cache not found: {tp}. "
            "Run V0.5 Winner vs Loser DNA once; do not re-download market data."
        )

    if progress_cb:
        progress_cb(5, "读取 V0.5 DNA Trade Cache")

    df = pd.read_csv(tp)
    required = {
        "symbol", "date", "pnl_r", "exit_reason",
        "prior_3bar_return_pct", "or_range_pct"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DNA cache schema missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    gated = _apply_gate(df)

    if progress_cb:
        progress_cb(18, "总体 Frozen Gate 验证")

    overall = _compare(df, gated)

    # Four fixed chronological blocks across the observed range.
    if progress_cb:
        progress_cb(30, "时间稳定性：4 个固定时间块")

    dmin, dmax = df["date"].min(), df["date"].max()
    span = max((dmax - dmin).days + 1, 1)
    time_rows = []
    for i in range(4):
        start = dmin + pd.Timedelta(days=int(span * i / 4))
        end = dmin + pd.Timedelta(days=int(span * (i + 1) / 4))
        if i == 3:
            mask = (df["date"] >= start) & (df["date"] <= dmax)
        else:
            mask = (df["date"] >= start) & (df["date"] < end)
        b = df[mask].copy()
        g = _apply_gate(b)
        row = _compare(b, g)
        row.update({
            "block": f"T{i+1}",
            "start": str(start.date()),
            "end": str((dmax if i == 3 else end - pd.Timedelta(days=1)).date()),
        })
        time_rows.append(row)

    # Four deterministic symbol buckets: fixed and reproducible, no cherry-picking.
    if progress_cb:
        progress_cb(48, "横截面稳定性：4 个固定 Symbol Buckets")

    df["_bucket"] = df["symbol"].map(lambda x: _stable_bucket(x, 4))
    bucket_rows = []
    for i in range(4):
        b = df[df["_bucket"] == i].copy()
        g = _apply_gate(b)
        row = _compare(b, g)
        row.update({"bucket": f"B{i+1}", "symbols": int(b["symbol"].nunique())})
        bucket_rows.append(row)

    if progress_cb:
        progress_cb(68, "Bootstrap 置信区间（固定种子，不调参）")

    boot = _bootstrap_gate(gated)

    if progress_cb:
        progress_cb(86, "执行预先定义的 Robustness Verdict")

    valid_time = [r for r in time_rows if r["gate"]["trades"] >= 5]
    valid_bucket = [r for r in bucket_rows if r["gate"]["trades"] >= 5]
    time_pf_gt1 = sum(r["gate"]["profit_factor"] > 1.0 for r in valid_time)
    bucket_pf_gt1 = sum(r["gate"]["profit_factor"] > 1.0 for r in valid_bucket)
    time_avg_gt0 = sum(r["gate"]["avg_r"] > 0.0 for r in valid_time)
    bucket_avg_gt0 = sum(r["gate"]["avg_r"] > 0.0 for r in valid_bucket)

    checks = {
        "overall_pf_ge_1_40": overall["gate"]["profit_factor"] >= 1.40,
        "overall_avg_r_ge_0_20": overall["gate"]["avg_r"] >= 0.20,
        "overall_trades_ge_40": overall["gate"]["trades"] >= 40,
        "time_blocks_pf_gt1_at_least_3_of_4":
            len(valid_time) >= 3 and time_pf_gt1 >= min(3, len(valid_time)),
        "time_blocks_avg_r_gt0_at_least_3_of_4":
            len(valid_time) >= 3 and time_avg_gt0 >= min(3, len(valid_time)),
        "symbol_buckets_pf_gt1_at_least_3_of_4":
            len(valid_bucket) >= 3 and bucket_pf_gt1 >= min(3, len(valid_bucket)),
        "symbol_buckets_avg_r_gt0_at_least_3_of_4":
            len(valid_bucket) >= 3 and bucket_avg_gt0 >= min(3, len(valid_bucket)),
        "bootstrap_prob_avg_r_gt0_ge_90pct":
            boot["prob_avg_r_gt0"] >= 0.90,
        "bootstrap_prob_pf_gt1_ge_90pct":
            boot["prob_pf_gt1"] >= 0.90,
    }

    passed = sum(bool(v) for v in checks.values())
    total = len(checks)

    if passed == total:
        verdict = "ROBUST_PASS"
    elif passed >= total - 2:
        verdict = "PROMISING_NOT_CONFIRMED"
    else:
        verdict = "ROBUST_FAIL"

    result = {
        "study": "V0.7 Robustness Validation",
        "gate": "prior 3-bar return <= 0.75% AND OR range >= 3.50%",
        "threshold_sweep": False,
        "same_sample_warning": (
            "This is robustness validation on the same historical year that revealed the DNA. "
            "It is NOT true untouched OOS. True confirmation still requires a later unseen period."
        ),
        "overall": overall,
        "time_blocks": time_rows,
        "symbol_buckets": bucket_rows,
        "bootstrap": boot,
        "checks": checks,
        "checks_passed": int(passed),
        "checks_total": int(total),
        "verdict": verdict,
        "next_step": (
            "If ROBUST_PASS: freeze the gate and wait for / test a truly unseen future period. "
            "If PROMISING_NOT_CONFIRMED: keep as research candidate only. "
            "If ROBUST_FAIL: do not tune the thresholds; return to diagnosis."
        ),
    }

    out = Path(outdir) if outdir else tp.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_v07.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    flat = []
    for r in time_rows:
        flat.append({
            "slice_type": "time", "slice": r["block"], "meta": f"{r['start']}..{r['end']}",
            "baseline_trades": r["baseline"]["trades"], "gate_trades": r["gate"]["trades"],
            "retention_pct": r["retention_pct"], "gate_pf": r["gate"]["profit_factor"],
            "gate_avg_r": r["gate"]["avg_r"], "gate_win_rate": r["gate"]["win_rate"],
            "gate_stop_rate": r["gate"]["stop_rate"],
        })
    for r in bucket_rows:
        flat.append({
            "slice_type": "symbol_bucket", "slice": r["bucket"], "meta": f"{r['symbols']} symbols",
            "baseline_trades": r["baseline"]["trades"], "gate_trades": r["gate"]["trades"],
            "retention_pct": r["retention_pct"], "gate_pf": r["gate"]["profit_factor"],
            "gate_avg_r": r["gate"]["avg_r"], "gate_win_rate": r["gate"]["win_rate"],
            "gate_stop_rate": r["gate"]["stop_rate"],
        })
    pd.DataFrame(flat).to_csv(out / "robustness_v07_slices.csv", index=False)

    if progress_cb:
        progress_cb(100, f"V0.7 完成：{verdict} ({passed}/{total} checks)")
    return result
