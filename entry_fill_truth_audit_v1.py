
import json
from pathlib import Path
import numpy as np
import pandas as pd

BT_DIR = Path("/data/daytrade/backtrader_parity_v2")
BT_TRADES = BT_DIR / "trade_results.csv"
PARITY24_DIR = Path("/data/daytrade/trade_level_parity_audit_v1")
PARITY24_OUTSIDE = PARITY24_DIR / "outside_21_bounds_diagnosed.csv"
CACHE_ROOT = Path("/data/daytrade/qc_1min_oos_cache")
OUTDIR = Path("/data/daytrade/entry_fill_truth_audit_v1")

REQ_CACHE = {"timestamp","open","high","low","close","volume"}

def _load_cache(path):
    x = pd.read_csv(path)
    missing = sorted(REQ_CACHE - set(x.columns))
    if missing:
        raise ValueError(f"cache schema missing {missing}; columns={list(x.columns)}")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").copy()
    if x.empty:
        raise ValueError("cache empty after parsing")
    if x.duplicated("timestamp").any():
        raise ValueError("duplicate 1-minute timestamps")
    return x

def _truth_row(r, bars):
    side = str(r["side"]).upper()
    intended = float(r["entry"])
    bt_fill = float(r["bt_entry_fill"])
    signal_time = pd.to_datetime(r["entry_time"], errors="coerce")
    if pd.isna(signal_time):
        return {"truth_status":"INVALID_SIGNAL_TIME"}

    win = bars[(bars["timestamp"] >= signal_time - pd.Timedelta(minutes=2)) &
               (bars["timestamp"] <= signal_time + pd.Timedelta(minutes=7))].copy()
    if win.empty:
        return {"truth_status":"NO_WINDOW"}

    activation = bars[bars["timestamp"] >= signal_time].copy()
    if activation.empty:
        return {"truth_status":"NO_ACTIVATION_BARS"}

    hit = None
    for b in activation.itertuples(index=False):
        touched = (float(b.high) >= intended) if side == "LONG" else (float(b.low) <= intended)
        if touched:
            hit = b
            break

    if hit is None:
        return {
            "truth_status":"NO_LEGAL_ENTRY_TOUCH",
            "legal_fill":False,
            "first_legal_touch_time":None,
            "first_legal_touch_open":np.nan,
            "expected_stop_fill":np.nan
        }

    if side == "LONG":
        expected = max(float(hit.open), intended)
    else:
        expected = min(float(hit.open), intended)

    tol = max(0.011, abs(expected) * 1e-8)
    legal = abs(bt_fill - expected) <= tol

    return {
        "truth_status":"PASS" if legal else "FAIL",
        "legal_fill":bool(legal),
        "likely_reason":"MATCHES_STOP_ORDER_SEMANTICS" if legal else "BT_FILL_DOES_NOT_MATCH_FIRST_LEGAL_STOP_FILL",
        "signal_time":str(signal_time),
        "first_legal_touch_time":str(hit.timestamp),
        "first_legal_touch_open":float(hit.open),
        "first_legal_touch_high":float(hit.high),
        "first_legal_touch_low":float(hit.low),
        "expected_stop_fill":float(expected),
        "bt_entry_fill":float(bt_fill),
        "fill_diff":float(bt_fill - expected),
        "window_json":json.dumps([
            {
                "timestamp":str(b.timestamp),
                "open":float(b.open),"high":float(b.high),
                "low":float(b.low),"close":float(b.close)
            }
            for b in win.itertuples(index=False)
        ], ensure_ascii=False)
    }

def run_entry_fill_truth_audit(progress_cb=None):
    if not PARITY24_OUTSIDE.exists():
        raise FileNotFoundError("24 outside_21_bounds_diagnosed.csv missing; run 24 first.")
    if not BT_TRADES.exists():
        raise FileNotFoundError("23 trade_results.csv missing; run 23 first.")

    outside = pd.read_csv(PARITY24_OUTSIDE)
    subset = outside[outside["likely_reason"].eq("ENTRY_PRICE_DIFFERENCE")].copy()

    if progress_cb:
        progress_cb(5, f"Preflight: ENTRY_PRICE_DIFFERENCE trades {len(subset)}")

    required = {"row_pos","symbol","date","side","entry_time","entry","bt_entry_fill"}
    missing = sorted(required - set(subset.columns))
    if missing:
        raise ValueError(f"24 output missing columns: {missing}")

    preflight = []
    for rr in subset.head(5).to_dict("records"):
        p = CACHE_ROOT / str(rr["symbol"]) / f"{rr['date']}.csv"
        item = {"symbol":rr["symbol"],"date":rr["date"],"path":str(p),"exists":p.exists()}
        if p.exists():
            try:
                x = _load_cache(p)
                item.update({
                    "rows":int(len(x)),
                    "columns":list(x.columns),
                    "time_min":str(x["timestamp"].min()),
                    "time_max":str(x["timestamp"].max())
                })
            except Exception as e:
                item["error"] = f"{type(e).__name__}: {e}"
        preflight.append(item)

    bad_preflight = [x for x in preflight if (not x.get("exists")) or x.get("error")]
    if bad_preflight:
        return {
            "study":"Entry Fill Truth Audit v1",
            "verdict":"PREFLIGHT_FAIL",
            "entry_diff_trades":int(len(subset)),
            "preflight":preflight,
            "decision":"STOP_BEFORE_AUDIT"
        }

    if progress_cb:
        progress_cb(15, "Preflight PASS · auditing first legal 1-minute stop fills")

    rows = []
    total = max(len(subset),1)
    for i, rr in enumerate(subset.to_dict("records"),1):
        p = CACHE_ROOT / str(rr["symbol"]) / f"{rr['date']}.csv"
        try:
            bars = _load_cache(p)
            truth = _truth_row(rr, bars)
        except Exception as e:
            truth = {"truth_status":"ERROR","legal_fill":False,"error":f"{type(e).__name__}: {e}"}

        rows.append({
            "row_pos":int(rr["row_pos"]),
            "symbol":str(rr["symbol"]),
            "date":str(rr["date"]),
            "side":str(rr["side"]),
            "entry_time":str(rr["entry_time"]),
            "intended_entry":float(rr["entry"]),
            "bt_entry_fill":float(rr["bt_entry_fill"]),
            **truth
        })

        if progress_cb and (i % 5 == 0 or i == total):
            ok = sum(1 for x in rows if x.get("truth_status")=="PASS")
            fail = sum(1 for x in rows if x.get("truth_status")=="FAIL")
            progress_cb(15 + int(75*i/total), f"Entry-fill truth {i}/{total} · PASS {ok} · FAIL {fail}")

    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR/"entry_fill_truth_rows.csv", index=False)

    counts = df["truth_status"].value_counts(dropna=False).to_dict() if not df.empty else {}
    pass_count = int(counts.get("PASS",0))
    fail_count = int(counts.get("FAIL",0))
    error_count = int(len(df) - pass_count - fail_count)

    mism = df[df["truth_status"].eq("FAIL")].copy()
    if not mism.empty and "fill_diff" in mism.columns:
        mism["abs_fill_diff"] = pd.to_numeric(mism["fill_diff"], errors="coerce").abs()
        largest = mism.nlargest(15,"abs_fill_diff")[
            ["row_pos","symbol","date","side","signal_time","first_legal_touch_time",
             "intended_entry","first_legal_touch_open","expected_stop_fill","bt_entry_fill","fill_diff"]
        ].to_dict("records")
    else:
        largest = []

    pass_share = pass_count/len(df) if len(df) else 0.0
    verdict = "PASS" if (len(df)>0 and pass_share >= 0.95 and error_count==0) else "FAIL"

    result = {
        "study":"Entry Fill Truth Audit v1",
        "strategy_changes":"NONE",
        "entry_diff_trades":int(len(df)),
        "pass_count":pass_count,
        "fail_count":fail_count,
        "error_count":error_count,
        "pass_share":round(float(pass_share),4),
        "verdict":verdict,
        "preflight":preflight,
        "largest_fill_mismatches":largest,
        "interpretation":(
            "PASS means >=95% of Backtrader entry-price differences match the first legal 1-minute stop-order fill "
            "(gap-through fills at bar open, otherwise at stop price). In that case Backtrader v23 is accepted as "
            "the primary execution benchmark. FAIL means execution implementation must be corrected before model conclusions."
        ),
        "next_step":"PAPER_FORWARD_TEST" if verdict=="PASS" else "FIX_EXECUTION_ONLY"
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")

    if progress_cb:
        progress_cb(100, f"Entry Fill Truth Audit · {verdict} · PASS {pass_count}/{len(df)}")
    return result
