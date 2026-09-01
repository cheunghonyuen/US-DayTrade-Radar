
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

API_ROOT = "https://api.tiingo.com/tiingo/equity/intraday"
CACHE_ROOT = Path("/data/daytrade/qc_1min_oos_cache")
STATE_FILE = CACHE_ROOT / "state.json"
RESULT_ROOT = Path("/data/daytrade/qc_1min_oos_result")

def _pf(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    return gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

def _summary(df):
    if df.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"median_r":0.0}
    r = pd.to_numeric(df["pnl_r"],errors="coerce").fillna(0.0)
    return {
        "trades":int(len(df)),
        "win_rate":round(float((r>0).mean()),4),
        "profit_factor":round(float(_pf(r)),4),
        "avg_r":round(float(r.mean()),4),
        "median_r":round(float(r.median()),4),
    }

def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed":[],"failed":[]}

def _save_state(st):
    CACHE_ROOT.mkdir(parents=True,exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(STATE_FILE)

def _cache_path(symbol,date):
    return CACHE_ROOT / symbol / f"{date}.csv"

def _download_1m(symbol,date,token,session):
    p=_cache_path(symbol,date)
    if p.exists():
        return p, "CACHE"
    p.parent.mkdir(parents=True,exist_ok=True)

    url=f"{API_ROOT}/{symbol}/prices"
    params={
        "startDate":date,
        "endDate":date,
        "resampleFreq":"1min",
        "columns":"open,high,low,close,volume",
        "token":token
    }
    last=None
    for attempt in range(4):
        try:
            resp=session.get(url,params=params,timeout=45)
            if resp.status_code==429:
                time.sleep(2.0*(attempt+1)); continue
            resp.raise_for_status()
            data=resp.json()
            if not isinstance(data,list):
                raise ValueError(f"unexpected response type {type(data)}")
            if not data:
                raise ValueError("empty 1min response")
            df=pd.DataFrame(data)
            if "date" not in df.columns:
                raise ValueError(f"missing date column: {list(df.columns)}")
            ts=pd.to_datetime(df["date"],errors="coerce",utc=True)
            if ts.isna().all():
                raise ValueError("all timestamps invalid")
            df["timestamp"]=ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
            for c in ["open","high","low","close","volume"]:
                if c in df.columns:
                    df[c]=pd.to_numeric(df[c],errors="coerce")
            df=df.dropna(subset=["timestamp","open","high","low","close"]).copy()
            df["time"]=df["timestamp"].dt.strftime("%H:%M")
            df=df[(df["time"]>="09:30")&(df["time"]<="15:59")].copy()
            if df.empty:
                raise ValueError("no regular-session 1min rows")
            df=df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp")
            tmp=p.with_suffix(".tmp")
            df.to_csv(tmp,index=False)
            tmp.replace(p)
            return p, "DOWNLOADED"
        except Exception as e:
            last=e
            time.sleep(1.0*(attempt+1))
    raise RuntimeError(f"{symbol} {date}: {last}")

def _minute_resolution(row, bars):
    side=1 if str(row.side).upper()=="LONG" else -1
    entry=float(row.entry); stop=float(row.stop)
    risk=abs(entry-stop)
    if risk<=0:
        return {"status":"INVALID","low_r":np.nan,"high_r":np.nan}

    bt=pd.to_datetime(row.entry_time,errors="coerce")
    if pd.isna(bt):
        return {"status":"INVALID_TIME","low_r":np.nan,"high_r":np.nan}

    # 5-min bar timestamp is the bar containing the breakout; inspect that 5-minute window.
    start=bt
    end=bt+pd.Timedelta(minutes=5)
    x=bars[(bars["timestamp"]>=start)&(bars["timestamp"]<end)].copy()
    if x.empty:
        # Some feeds timestamp bars at end-of-minute. Broaden by one minute on both sides.
        x=bars[(bars["timestamp"]>=start-pd.Timedelta(minutes=1))&
               (bars["timestamp"]<end+pd.Timedelta(minutes=1))].copy()
    if x.empty:
        return {"status":"NO_1MIN_WINDOW","low_r":np.nan,"high_r":np.nan}

    hit_idx=None
    same_minute_ambiguous=False
    for idx,b in x.iterrows():
        entry_hit=(float(b.high)>=entry) if side==1 else (float(b.low)<=entry)
        if not entry_hit:
            continue
        hit_idx=idx
        stop_hit=(float(b.low)<=stop) if side==1 else (float(b.high)>=stop)
        same_minute_ambiguous=bool(stop_hit)
        break
    if hit_idx is None:
        return {"status":"NO_ENTRY_IN_1MIN","low_r":np.nan,"high_r":np.nan}

    # Conservative lower bound if entry and stop share the same 1-minute bar.
    if same_minute_ambiguous:
        low_r=-1.0
        later=bars[bars.index>hit_idx].copy()
        exit_px=None
        for _,b in later.iterrows():
            stop_hit=(float(b.low)<=stop) if side==1 else (float(b.high)>=stop)
            if stop_hit:
                exit_px=stop; break
        if exit_px is None:
            exit_px=float(bars.iloc[-1]["close"])
        high_r=(exit_px-entry)*side/risk
        return {"status":"UNRESOLVED_SAME_1MIN","low_r":float(low_r),"high_r":float(high_r)}

    # Entry occurred in a minute that did not touch stop; sequence is resolved from next minutes onward.
    later=bars[bars.index>hit_idx].copy()
    exit_px=None
    reason="EOD"
    for _,b in later.iterrows():
        stop_hit=(float(b.low)<=stop) if side==1 else (float(b.high)>=stop)
        if stop_hit:
            exit_px=stop; reason="STOP"; break
    if exit_px is None:
        exit_px=float(bars.iloc[-1]["close"])
    r=(exit_px-entry)*side/risk
    return {"status":"RESOLVED","low_r":float(r),"high_r":float(r),"reason":reason}

def run_qc_1min_oos_resolver(qc_result_dir, progress_cb=None):
    qdir=Path(qc_result_dir)
    strict_path=qdir/"qc_oos_strict.csv"
    delayed_path=qdir/"qc_oos_delayed_stop.csv"
    if not strict_path.exists() or not delayed_path.exists():
        raise FileNotFoundError("Run QuantConnect External Benchmark v1 first.")

    token=os.getenv("TIINGO_TOKEN")
    if not token:
        raise RuntimeError("TIINGO_TOKEN missing")

    strict=pd.read_csv(strict_path)
    delayed=pd.read_csv(delayed_path)
    if len(strict)!=len(delayed):
        raise ValueError("strict/delayed row count mismatch")

    # Same trade identity and order expected from benchmark.
    keys=["symbol","date","side","entry_time","entry","stop"]
    for k in keys:
        if k not in strict.columns or k not in delayed.columns:
            raise ValueError(f"missing key {k}")

    same_key=(strict[keys].astype(str).values==delayed[keys].astype(str).values).all()
    if not same_key:
        raise ValueError("strict/delayed trade identity mismatch")

    diff=(pd.to_numeric(strict["pnl_r"],errors="coerce").round(6) !=
          pd.to_numeric(delayed["pnl_r"],errors="coerce").round(6))
    ambiguous_idx=np.flatnonzero(diff.to_numpy())
    stable_idx=np.flatnonzero((~diff).to_numpy())

    if progress_cb:
        progress_cb(3,f"OOS trades {len(strict)} · 5min ambiguity {len(ambiguous_idx)}")

    state=_load_state()
    completed=set(state.get("completed",[]))
    failed=list(state.get("failed",[]))
    session=requests.Session()
    session.headers.update({"Content-Type":"application/json"})

    rows=[]
    total=max(len(ambiguous_idx),1)
    for n,pos in enumerate(ambiguous_idx,1):
        r=strict.iloc[pos]
        key=f"{r.symbol}|{r.date}"
        try:
            p,src=_download_1m(str(r.symbol),str(r.date),token,session)
            bars=pd.read_csv(p)
            bars["timestamp"]=pd.to_datetime(bars["timestamp"],errors="coerce")
            res=_minute_resolution(r,bars)
            rows.append({"row_pos":int(pos),"symbol":r.symbol,"date":r.date,
                         "cache_source":src,**res})
            completed.add(key)
        except Exception as e:
            rows.append({"row_pos":int(pos),"symbol":r.symbol,"date":r.date,
                         "status":"DOWNLOAD_ERROR","error":str(e),
                         "low_r":np.nan,"high_r":np.nan})
            failed.append({"key":key,"error":str(e)})
        if n%10==0 or n==total:
            state={"completed":sorted(completed),"failed":failed[-200:]}
            _save_state(state)
            if progress_cb:
                progress_cb(5+int(85*n/total),f"1min resolve {n}/{total} · cache {len(completed)}")

    resolved=pd.DataFrame(rows)
    RESULT_ROOT.mkdir(parents=True,exist_ok=True)
    resolved.to_csv(RESULT_ROOT/"qc_oos_1min_resolution.csv",index=False)

    # Build final lower/upper vectors:
    # stable 5-min trades are identical in both bounds, so no finer data needed.
    low=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    high=pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    for rr in rows:
        pos=int(rr["row_pos"])
        if np.isfinite(rr.get("low_r",np.nan)):
            low.iloc[pos]=float(rr["low_r"])
        if np.isfinite(rr.get("high_r",np.nan)):
            high.iloc[pos]=float(rr["high_r"])
        else:
            # unresolved download error falls back to original 5-min upper bound
            high.iloc[pos]=float(delayed.iloc[pos]["pnl_r"])

    low_df=strict.copy(); low_df["pnl_r"]=low
    high_df=strict.copy(); high_df["pnl_r"]=high

    status_counts=resolved["status"].value_counts(dropna=False).to_dict() if not resolved.empty else {}
    result={
        "study":"QuantConnect ORB · True OOS 1-Minute Ambiguity Resolver v1",
        "oos_trades":int(len(strict)),
        "five_min_ambiguous_trades":int(len(ambiguous_idx)),
        "five_min_stable_trades":int(len(stable_idx)),
        "resolution_status":{str(k):int(v) for k,v in status_counts.items()},
        "lower_bound":_summary(low_df),
        "upper_bound":_summary(high_df),
        "remaining_1min_ambiguity":int(status_counts.get("UNRESOLVED_SAME_1MIN",0)),
        "download_errors":int(status_counts.get("DOWNLOAD_ERROR",0)),
        "cache_dir":str(CACHE_ROOT),
        "checkpoint_file":str(STATE_FILE),
        "decision_rule":"If 1-min lower and upper OOS bounds are both PF>1 with AvgR>0, promote to Backtrader/LEAN parity. If both PF<1 or AvgR<0, DROP. If the bounds still straddle profitability, only tick/event-order data can decide; do not tune strategy parameters."
    }
    (RESULT_ROOT/"qc_oos_1min_resolver_v1.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    if progress_cb:
        progress_cb(100,f"1min resolver complete · remaining ambiguous {result['remaining_1min_ambiguity']}")
    return result
