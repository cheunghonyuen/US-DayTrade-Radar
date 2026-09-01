
import json
from pathlib import Path
import numpy as np
import pandas as pd

BT_DIR = Path("/data/daytrade/backtrader_parity_v2")
BT_RESULT = BT_DIR / "result.json"
BT_TRADES = BT_DIR / "trade_results.csv"

QC_DIR_CANDIDATES = [
    Path("/data/daytrade/outputs/qc_stocks_in_play_benchmark_v1"),
    Path("/data/daytrade/results/qc_stocks_in_play_benchmark_v1"),
    Path("/data/daytrade/qc_stocks_in_play_benchmark_v1"),
]
RESOLUTION_FILE = Path("/data/daytrade/qc_1min_oos_result/qc_oos_1min_resolution.csv")
CACHE_ROOT = Path("/data/daytrade/qc_1min_oos_cache")
OUTDIR = Path("/data/daytrade/trade_level_parity_audit_v1")

def _locate_qdir():
    for p in QC_DIR_CANDIDATES:
        if (p/"qc_oos_strict.csv").exists() and (p/"qc_oos_delayed_stop.csv").exists():
            return p
    hits=list(Path("/data/daytrade").rglob("qc_oos_strict.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError("qc_oos_strict.csv not found")

def _load():
    if not BT_RESULT.exists() or not BT_TRADES.exists():
        raise FileNotFoundError("23 outputs missing; run Backtrader Parity v2 first.")
    qdir=_locate_qdir()
    strict=pd.read_csv(qdir/"qc_oos_strict.csv")
    delayed=pd.read_csv(qdir/"qc_oos_delayed_stop.csv")
    bt=pd.read_csv(BT_TRADES)
    res=pd.read_csv(RESOLUTION_FILE) if RESOLUTION_FILE.exists() else pd.DataFrame()
    return strict, delayed, bt, res

def _build_bounds(strict, delayed, res):
    lower=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    upper=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    if not res.empty:
        for rr in res.to_dict("records"):
            pos=int(rr["row_pos"])
            lo=pd.to_numeric(pd.Series([rr.get("low_r")]),errors="coerce").iloc[0]
            hi=pd.to_numeric(pd.Series([rr.get("high_r")]),errors="coerce").iloc[0]
            if pd.notna(lo):
                lower.iloc[pos]=float(lo)
            if pd.notna(hi):
                upper.iloc[pos]=float(hi)
            else:
                upper.iloc[pos]=float(delayed.iloc[pos]["pnl_r"])
    return lower, upper

def _classify_trade(base_row, bt_row, lo, hi):
    br=pd.to_numeric(pd.Series([bt_row.get("pnl_r")]),errors="coerce").iloc[0]
    if pd.isna(br):
        return "BT_NO_RESULT", np.nan
    d=0.0
    low=min(lo,hi); high=max(lo,hi)
    if br < low:
        d=float(br-low)
        cls="BT_BELOW_21_LOWER"
    elif br > high:
        d=float(br-high)
        cls="BT_ABOVE_21_UPPER"
    else:
        cls="INSIDE_21_BOUNDS"
    return cls,d

def _inspect_1m(row, cache_path):
    x=pd.read_csv(cache_path)
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
    x=x.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    et=pd.to_datetime(row["entry_time"],errors="coerce")
    if pd.isna(et):
        return {}
    w=x[(x["timestamp"]>=et-pd.Timedelta(minutes=2)) &
        (x["timestamp"]<=et+pd.Timedelta(minutes=7))].copy()
    if w.empty:
        return {}
    side=str(row["side"]).upper()
    entry=float(row["entry"]); stop=float(row["stop"])
    detail=[]
    for b in w.itertuples(index=False):
        eh=(float(b.high)>=entry) if side=="LONG" else (float(b.low)<=entry)
        sh=(float(b.low)<=stop) if side=="LONG" else (float(b.high)>=stop)
        detail.append({
            "timestamp":str(b.timestamp),
            "open":float(b.open),"high":float(b.high),"low":float(b.low),"close":float(b.close),
            "entry_hit":bool(eh),"stop_hit":bool(sh)
        })
    return {"window":detail}

def run_trade_level_parity_audit(progress_cb=None):
    strict, delayed, bt, res=_load()
    lower, upper=_build_bounds(strict, delayed, res)

    if progress_cb:
        progress_cb(5,"读取 23 trade_results + 21 lower/upper；不重跑策略")

    # Join using row_pos only; this is the stable positional identity used in 21/23.
    bt_ok=bt[bt["status"].eq("OK")].copy()
    rows=[]
    outside=[]
    total=max(len(bt_ok),1)

    for n,rr in enumerate(bt_ok.to_dict("records"),1):
        pos=int(rr["row_pos"])
        base=strict.iloc[pos]
        lo=float(lower.iloc[pos]); hi=float(upper.iloc[pos])
        cls,delta=_classify_trade(base,rr,lo,hi)

        rec={
            "row_pos":pos,
            "symbol":str(base["symbol"]),
            "date":str(base["date"]),
            "side":str(base["side"]),
            "entry_time":str(base["entry_time"]),
            "entry":float(base["entry"]),
            "stop":float(base["stop"]),
            "strict_r":float(strict.iloc[pos]["pnl_r"]),
            "delayed_r":float(delayed.iloc[pos]["pnl_r"]),
            "lower_21_r":lo,
            "upper_21_r":hi,
            "backtrader_r":float(rr["pnl_r"]),
            "bt_entry_fill":float(rr["entry_fill"]) if pd.notna(rr.get("entry_fill")) else np.nan,
            "bt_exit_fill":float(rr["exit_fill"]) if pd.notna(rr.get("exit_fill")) else np.nan,
            "bt_exit_reason":str(rr.get("exit_reason","")),
            "classification":cls,
            "distance_outside":float(delta) if np.isfinite(delta) else np.nan,
        }
        rows.append(rec)
        if cls!="INSIDE_21_BOUNDS":
            p=CACHE_ROOT/rec["symbol"]/f"{rec['date']}.csv"
            rec["cache_exists"]=p.exists()
            if p.exists():
                try:
                    rec["minute_window"]=json.dumps(_inspect_1m(base,p),ensure_ascii=False)
                except Exception as e:
                    rec["minute_window_error"]=f"{type(e).__name__}: {e}"
            outside.append(rec.copy())

        if progress_cb and (n%25==0 or n==total):
            progress_cb(5+int(70*n/total),f"逐笔对账 {n}/{total} · outside {len(outside)}")

    all_df=pd.DataFrame(rows)
    out_df=pd.DataFrame(outside)
    OUTDIR.mkdir(parents=True,exist_ok=True)
    all_df.to_csv(OUTDIR/"all_ambiguous_trade_parity.csv",index=False)
    out_df.to_csv(OUTDIR/"outside_21_bounds.csv",index=False)

    if progress_cb:
        progress_cb(82,"归因 outside-bound trades")

    counts=all_df["classification"].value_counts().to_dict() if not all_df.empty else {}
    below=out_df[out_df["classification"].eq("BT_BELOW_21_LOWER")] if not out_df.empty else out_df
    above=out_df[out_df["classification"].eq("BT_ABOVE_21_UPPER")] if not out_df.empty else out_df

    # Heuristic diagnosis from exact recorded fills vs intended levels.
    # This is not strategy tuning; it only labels likely execution reasons.
    reasons={"ENTRY_PRICE_DIFFERENCE":0,"STOP_OR_EXIT_SEQUENCE":0,"OTHER":0}
    diagnosed=[]
    for rr in outside:
        r=rr.copy()
        entry_diff=np.isfinite(r.get("bt_entry_fill",np.nan)) and abs(float(r["bt_entry_fill"])-float(r["entry"]))>1e-8
        if entry_diff:
            reason="ENTRY_PRICE_DIFFERENCE"
        elif str(r.get("bt_exit_reason",""))=="STOP":
            reason="STOP_OR_EXIT_SEQUENCE"
        else:
            reason="OTHER"
        reasons[reason]+=1
        r["likely_reason"]=reason
        diagnosed.append(r)

    diag_df=pd.DataFrame(diagnosed)
    diag_df.to_csv(OUTDIR/"outside_21_bounds_diagnosed.csv",index=False)

    result={
        "study":"Trade-Level Parity Audit v1",
        "source":"21 one-minute lower/upper + 23 Backtrader v2 executed trades",
        "strategy_changes":"NONE",
        "ambiguous_trades_checked":int(len(all_df)),
        "inside_bounds":int(counts.get("INSIDE_21_BOUNDS",0)),
        "outside_bounds":int(len(out_df)),
        "inside_share":round(float(counts.get("INSIDE_21_BOUNDS",0)/len(all_df)),4) if len(all_df) else 0.0,
        "below_lower":int(len(below)),
        "above_upper":int(len(above)),
        "likely_reason_counts":reasons,
        "largest_negative_outside":(
            out_df.nsmallest(10,"distance_outside")[
                ["row_pos","symbol","date","side","lower_21_r","upper_21_r","backtrader_r","distance_outside","bt_entry_fill","bt_exit_fill","bt_exit_reason"]
            ].to_dict("records") if not out_df.empty else []
        ),
        "largest_positive_outside":(
            out_df.nlargest(10,"distance_outside")[
                ["row_pos","symbol","date","side","lower_21_r","upper_21_r","backtrader_r","distance_outside","bt_entry_fill","bt_exit_fill","bt_exit_reason"]
            ].to_dict("records") if not out_df.empty else []
        ),
        "decision_rule":"No strategy tuning. If outside-bound differences are explained by execution-model semantics while aggregate Backtrader PF/AvgR stay positive, accept Backtrader as primary execution benchmark. If systematic impossible fills or timestamp errors appear, execution implementation must be fixed before any model conclusion."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Trade-level audit complete · outside {result['outside_bounds']} / {result['ambiguous_trades_checked']}")
    return result
