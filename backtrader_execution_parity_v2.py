
import json
from pathlib import Path
import numpy as np
import pandas as pd
import backtrader as bt

QC_DIR_CANDIDATES = [
    Path("/data/daytrade/outputs/qc_stocks_in_play_benchmark_v1"),
    Path("/data/daytrade/results/qc_stocks_in_play_benchmark_v1"),
    Path("/data/daytrade/qc_stocks_in_play_benchmark_v1"),
]
RESOLUTION_FILE = Path("/data/daytrade/qc_1min_oos_result/qc_oos_1min_resolution.csv")
CACHE_ROOT = Path("/data/daytrade/qc_1min_oos_cache")
OUTDIR = Path("/data/daytrade/backtrader_parity_v2")

REQ_CACHE = {"timestamp","open","high","low","close","volume"}

def _pf(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    return gp/gl if gl>0 else (999.0 if gp>0 else 0.0)

def _summary_r(r):
    r=pd.Series(pd.to_numeric(r,errors="coerce")).dropna()
    if r.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"median_r":0.0}
    return {
        "trades":int(len(r)),
        "win_rate":round(float((r>0).mean()),4),
        "profit_factor":round(float(_pf(r)),4),
        "avg_r":round(float(r.mean()),4),
        "median_r":round(float(r.median()),4),
    }

def _locate_qdir():
    for p in QC_DIR_CANDIDATES:
        if (p/"qc_oos_strict.csv").exists() and (p/"qc_oos_delayed_stop.csv").exists():
            return p
    root=Path("/data/daytrade")
    hits=list(root.rglob("qc_oos_strict.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError("qc_oos_strict.csv not found")

def _cache_path(symbol,date):
    return CACHE_ROOT / str(symbol) / f"{str(date)}.csv"

def _load_cache(path):
    x=pd.read_csv(path)
    miss=sorted(REQ_CACHE-set(x.columns))
    if miss:
        raise ValueError(f"cache schema missing {miss}; columns={list(x.columns)}")
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
    for c in ["open","high","low","close","volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").copy()
    if x.empty:
        raise ValueError("cache has no valid rows")
    if x.duplicated("timestamp").any():
        raise ValueError("cache has duplicate timestamps")
    return x

class OneTrade(bt.Strategy):
    params = dict(side="LONG", entry=0.0, stop=0.0, activation=None)

    def __init__(self):
        self.submitted=False
        self.entry_order=None
        self.stop_order=None
        self.entry_fill=None
        self.exit_fill=None
        self.exit_reason=None
        self.order_log=[]

    def next(self):
        dt=self.data.datetime.datetime(0)
        # Submit on the bar immediately before activation. The broker can then
        # evaluate the stop-entry on the activation bar.
        if (not self.submitted) and dt >= self.p.activation - pd.Timedelta(minutes=1):
            self.submitted=True
            if self.p.side=="LONG":
                self.entry_order=self.buy(size=1, exectype=bt.Order.Stop, price=float(self.p.entry))
            else:
                self.entry_order=self.sell(size=1, exectype=bt.Order.Stop, price=float(self.p.entry))

    def notify_order(self, order):
        self.order_log.append({
            "ref":int(order.ref),
            "status":order.getstatusname(),
            "isbuy":bool(order.isbuy()),
            "price":float(order.executed.price or 0.0),
            "size":float(order.executed.size or 0.0),
        })
        if order.status != bt.Order.Completed:
            return
        if self.entry_order is not None and order.ref==self.entry_order.ref:
            self.entry_fill=float(order.executed.price)
            if self.p.side=="LONG":
                self.stop_order=self.sell(size=1, exectype=bt.Order.Stop, price=float(self.p.stop))
            else:
                self.stop_order=self.buy(size=1, exectype=bt.Order.Stop, price=float(self.p.stop))
        elif self.stop_order is not None and order.ref==self.stop_order.ref:
            self.exit_fill=float(order.executed.price)
            self.exit_reason="STOP"

    def stop(self):
        if self.entry_fill is not None and self.exit_fill is None:
            self.exit_fill=float(self.data.close[0])
            self.exit_reason="EOD"

def _run_bt(row,bars):
    entry_time=pd.Timestamp(pd.to_datetime(row["entry_time"],errors="raise")).to_pydatetime()
    side=str(row["side"]).upper()
    entry=float(row["entry"]); stop=float(row["stop"])
    risk=abs(entry-stop)
    if risk<=0:
        raise ValueError("non-positive risk")

    # Need one bar before entry_time so stop-entry is live for entry minute.
    start=pd.Timestamp(entry_time)-pd.Timedelta(minutes=1)
    x=bars[bars["timestamp"]>=start].copy()
    if len(x)<2:
        raise ValueError(f"insufficient feed rows from {start}: {len(x)}")

    f=x.set_index("timestamp")[["open","high","low","close","volume"]].copy()
    f["openinterest"]=0.0
    if not isinstance(f.index,pd.DatetimeIndex):
        raise TypeError(f"feed index type {type(f.index)}")

    cerebro=bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=0.0)
    data=bt.feeds.PandasData(
        dataname=f,
        datetime=None,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest="openinterest",
        timeframe=bt.TimeFrame.Minutes, compression=1,
    )
    cerebro.adddata(data)
    cerebro.addstrategy(
        OneTrade, side=side, entry=entry, stop=stop, activation=entry_time
    )
    st=cerebro.run(runonce=False,preload=True,maxcpus=1)[0]

    if st.entry_fill is None:
        return {"status":"NO_ENTRY","pnl_r":np.nan,"order_log":st.order_log}
    if st.exit_fill is None:
        return {"status":"NO_EXIT","pnl_r":np.nan,"order_log":st.order_log}

    direction=1.0 if side=="LONG" else -1.0
    pnl_r=(st.exit_fill-st.entry_fill)*direction/risk
    return {
        "status":"OK",
        "pnl_r":float(pnl_r),
        "entry_fill":float(st.entry_fill),
        "exit_fill":float(st.exit_fill),
        "exit_reason":st.exit_reason,
        "order_log":st.order_log,
    }

def _preflight(strict, delayed, ambiguous):
    report={
        "backtrader_version":getattr(bt,"__version__","unknown"),
        "strict_rows":int(len(strict)),
        "delayed_rows":int(len(delayed)),
        "ambiguous":int(len(ambiguous)),
        "resolution_exists":RESOLUTION_FILE.exists(),
        "cache_root_exists":CACHE_ROOT.exists(),
        "cache_files_total":int(len(list(CACHE_ROOT.rglob("*.csv")))) if CACHE_ROOT.exists() else 0,
        "sample_checks":[]
    }
    for pos in ambiguous[:5]:
        r=strict.iloc[int(pos)]
        p=_cache_path(r["symbol"],r["date"])
        item={"row_pos":int(pos),"symbol":str(r["symbol"]),"date":str(r["date"]),"path":str(p),"exists":p.exists()}
        if p.exists():
            try:
                x=_load_cache(p)
                item.update({
                    "rows":int(len(x)),
                    "columns":list(x.columns),
                    "time_min":str(x["timestamp"].min()),
                    "time_max":str(x["timestamp"].max()),
                })
                trial=_run_bt(r,x)
                item["trial_status"]=trial["status"]
                item["trial_pnl_r"]=None if not np.isfinite(trial.get("pnl_r",np.nan)) else round(float(trial["pnl_r"]),6)
                item["order_log"]=trial.get("order_log",[])
            except Exception as e:
                item["trial_error"]=f"{type(e).__name__}: {e}"
        report["sample_checks"].append(item)
    return report

def run_backtrader_parity_v2(progress_cb=None):
    qdir=_locate_qdir()
    strict=pd.read_csv(qdir/"qc_oos_strict.csv")
    delayed=pd.read_csv(qdir/"qc_oos_delayed_stop.csv")
    if len(strict)!=len(delayed):
        raise ValueError("strict/delayed row count mismatch")

    diff=(pd.to_numeric(strict["pnl_r"],errors="coerce").round(6) !=
          pd.to_numeric(delayed["pnl_r"],errors="coerce").round(6))
    ambiguous=np.flatnonzero(diff.to_numpy())

    if progress_cb:
        progress_cb(3,"Runtime/Schema Preflight on real 1-minute cache")

    pre=_preflight(strict,delayed,ambiguous)
    OUTDIR.mkdir(parents=True,exist_ok=True)
    (OUTDIR/"preflight.json").write_text(json.dumps(pre,indent=2,ensure_ascii=False),encoding="utf-8")

    sample_errors=[x for x in pre["sample_checks"] if x.get("trial_error") or not x.get("exists")]
    if sample_errors:
        return {
            "study":"Backtrader Independent Engine Parity v2",
            "verdict":"PREFLIGHT_FAIL",
            "preflight":pre,
            "backtrader_oos":_summary_r(pd.to_numeric(strict["pnl_r"],errors="coerce")),
            "resolved":0,
            "errors":len(ambiguous),
            "decision":"STOP_BEFORE_LONG_RUN",
            "note":"Real cache preflight failed on sample trades; no long run attempted."
        }

    if progress_cb:
        progress_cb(12,f"Preflight PASS · replay {len(ambiguous)} ambiguous trades")

    # Rebuild 21 lower/upper vectors.
    lower=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    upper=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    if RESOLUTION_FILE.exists():
        res=pd.read_csv(RESOLUTION_FILE)
        for rr in res.to_dict("records"):
            pos=int(rr["row_pos"])
            lo=pd.to_numeric(pd.Series([rr.get("low_r")]),errors="coerce").iloc[0]
            hi=pd.to_numeric(pd.Series([rr.get("high_r")]),errors="coerce").iloc[0]
            if pd.notna(lo): lower.iloc[pos]=float(lo)
            if pd.notna(hi): upper.iloc[pos]=float(hi)
            else: upper.iloc[pos]=float(delayed.iloc[pos]["pnl_r"])

    bt_vec=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    details=[]
    status_counts={}
    total=max(len(ambiguous),1)

    for n,pos in enumerate(ambiguous,1):
        r=strict.iloc[int(pos)]
        p=_cache_path(r["symbol"],r["date"])
        try:
            if not p.exists():
                raise FileNotFoundError(str(p))
            bars=_load_cache(p)
            out=_run_bt(r,bars)
        except Exception as e:
            out={"status":"ERROR","pnl_r":np.nan,"error":f"{type(e).__name__}: {e}"}

        status=str(out.get("status","UNKNOWN"))
        status_counts[status]=status_counts.get(status,0)+1
        rec={
            "row_pos":int(pos),"symbol":str(r["symbol"]),"date":str(r["date"]),
            "status":status,"pnl_r":out.get("pnl_r",np.nan),
            "entry_fill":out.get("entry_fill",np.nan),
            "exit_fill":out.get("exit_fill",np.nan),
            "exit_reason":out.get("exit_reason",""),
            "error":out.get("error",""),
            "order_log_json":json.dumps(out.get("order_log",[]),ensure_ascii=False)
        }
        details.append(rec)
        if status=="OK" and np.isfinite(out.get("pnl_r",np.nan)):
            bt_vec.iloc[int(pos)]=float(out["pnl_r"])

        if progress_cb and (n%10==0 or n==total):
            progress_cb(12+int(80*n/total),f"Backtrader replay {n}/{total} · OK {status_counts.get('OK',0)} · ERR {status_counts.get('ERROR',0)}")

    detail=pd.DataFrame(details)
    detail.to_csv(OUTDIR/"trade_results.csv",index=False)

    bt_s=_summary_r(bt_vec); lo_s=_summary_r(lower); hi_s=_summary_r(upper)
    resolved=int(status_counts.get("OK",0))
    errors=int(len(ambiguous)-resolved)

    inside=0; compared=0
    for rr in details:
        if rr["status"]!="OK" or not np.isfinite(rr["pnl_r"]):
            continue
        pos=rr["row_pos"]; lo=float(lower.iloc[pos]); hi=float(upper.iloc[pos]); br=float(rr["pnl_r"])
        compared+=1
        if min(lo,hi)-1e-9<=br<=max(lo,hi)+1e-9:
            inside+=1

    pass_flag=bool(
        resolved>=int(0.95*len(ambiguous)) and
        bt_s["profit_factor"]>1 and bt_s["avg_r"]>0 and
        lo_s["profit_factor"]>1 and lo_s["avg_r"]>0 and
        hi_s["profit_factor"]>1 and hi_s["avg_r"]>0
    )

    result={
        "study":"Backtrader Independent Engine Parity v2",
        "verdict":"PASS" if pass_flag else "FAIL",
        "preflight":pre,
        "status_counts":status_counts,
        "resolved":resolved,
        "errors":errors,
        "oos_trades":int(len(strict)),
        "ambiguous_trades":int(len(ambiguous)),
        "lower_bound_21":lo_s,
        "backtrader_oos":bt_s,
        "upper_bound_21":hi_s,
        "inside_21_bounds_share":round(float(inside/compared),4) if compared else 0.0,
        "decision_rule":"PASS requires >=95% of ambiguous trades successfully replayed, Backtrader OOS PF>1 and AvgR>0, and 21 lower/upper both positive. No strategy tuning."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Backtrader v2 complete · resolved {resolved}/{len(ambiguous)} · PF {bt_s['profit_factor']}")
    return result
