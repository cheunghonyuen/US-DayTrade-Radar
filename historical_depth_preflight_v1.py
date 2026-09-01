
import os,json,time
from pathlib import Path
import pandas as pd, requests

ROOT=Path("/data/daytrade")
SIP=ROOT/"sip_data.csv"
OUT=ROOT/"formation_daily_v1"
PRE=OUT/"preflight.json"
MAN=OUT/"manifest.json"
API="https://api.tiingo.com/tiingo/daily"

def _symbols():
    x=pd.read_csv(SIP,usecols=["symbol","timestamp"])
    syms=sorted(x["symbol"].dropna().astype(str).unique())
    if len(syms)!=125: raise ValueError(f"expected 125 symbols, found {len(syms)}")
    ts=pd.to_datetime(x["timestamp"],errors="coerce",utc=True)
    return syms,{"rows":len(x),"symbols":len(syms),"min":str(ts.min()),"max":str(ts.max()),"bad_ts":int(ts.isna().sum())}

def _fetch(sym,start,end,token,sess):
    last=None
    for a in range(4):
        try:
            r=sess.get(f"{API}/{sym}/prices",params={"startDate":start,"endDate":end,"format":"json","token":token},timeout=45)
            if r.status_code==429:
                time.sleep(1.5*(a+1)); continue
            r.raise_for_status()
            d=pd.DataFrame(r.json())
            need={"date","open","high","low","close","volume"}
            miss=sorted(need-set(d.columns))
            if miss: raise ValueError(f"schema missing {miss}; cols={list(d.columns)}")
            d["date"]=pd.to_datetime(d["date"],errors="coerce",utc=True).dt.date.astype(str)
            for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
            keep=[c for c in ["date","open","high","low","close","volume","adjOpen","adjHigh","adjLow","adjClose","adjVolume","divCash","splitFactor"] if c in d.columns]
            d=d[keep].dropna(subset=["date","open","high","low","close"]).sort_values("date")
            if d.empty or d.duplicated("date").any(): raise ValueError("empty/duplicate daily data")
            return d
        except Exception as e:
            last=e; time.sleep(a+1)
    raise RuntimeError(f"{sym}: {last}")

def run_historical_depth_preflight(progress_cb=None):
    OUT.mkdir(parents=True,exist_ok=True)
    token=os.getenv("TIINGO_TOKEN")
    if not token: raise RuntimeError("TIINGO_TOKEN missing")
    syms,meta=_symbols()
    start,end="2021-01-01","2026-08-30"
    idx=[0,len(syms)//4,len(syms)//2,3*len(syms)//4,len(syms)-1]
    picks=[syms[i] for i in idx]
    sess=requests.Session(); samples=[]
    if progress_cb: progress_cb(10,f"Replay PASS · {len(syms)} symbols · test 5 daily samples")
    for i,s in enumerate(picks,1):
        d=_fetch(s,start,end,token,sess)
        samples.append({"symbol":s,"rows":len(d),"date_min":str(d.date.min()),"date_max":str(d.date.max()),"columns":list(d.columns)})
        if progress_cb: progress_cb(10+int(60*i/5),f"sample {i}/5 · {s} · {len(d)} rows")
    avg=sum(x["rows"] for x in samples)/len(samples)
    est_rows=round(avg*len(syms))
    est_mb=round(est_rows*110/1024/1024,1)
    r={"study":"Historical Depth Preflight v1","verdict":"PASS","persistent_replay":meta,
       "target":{"start":start,"end":end},"symbols":len(syms),"samples":samples,
       "estimated_rows":est_rows,"estimated_storage_mb":est_mb,"requests_needed":len(syms),
       "guardrail":"Build allowed only after PASS. Existing daily symbol cache is reused; intraday replay and Frozen Forward are untouched."}
    PRE.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb: progress_cb(100,f"PASS · est {est_rows:,} rows · ~{est_mb} MB")
    return r

def run_historical_depth_build(progress_cb=None):
    if not PRE.exists(): raise RuntimeError("Run preflight first")
    pre=json.loads(PRE.read_text(encoding="utf-8"))
    if pre.get("verdict")!="PASS": raise RuntimeError("Preflight not PASS")
    token=os.getenv("TIINGO_TOKEN")
    syms,_=_symbols(); sess=requests.Session()
    start,end=pre["target"]["start"],pre["target"]["end"]
    ready=[]; errors=[]; reused=downloaded=0
    for i,s in enumerate(syms,1):
        p=OUT/f"{s}.csv"
        try:
            ok=False
            if p.exists():
                old=pd.read_csv(p)
                ok={"date","open","high","low","close","volume"}.issubset(old.columns) and len(old)>500 and str(old.date.min())<=start and str(old.date.max())>=end and not old.duplicated("date").any()
            if ok:
                d=old; reused+=1
            else:
                d=_fetch(s,start,end,token,sess)
                tmp=p.with_suffix(".tmp"); d.to_csv(tmp,index=False); tmp.replace(p); downloaded+=1
            ready.append({"symbol":s,"rows":len(d),"date_min":str(d.date.min()),"date_max":str(d.date.max()),"path":str(p)})
        except Exception as e:
            errors.append({"symbol":s,"error":f"{type(e).__name__}: {e}"})
        if progress_cb and (i%5==0 or i==len(syms)):
            progress_cb(int(100*i/len(syms)),f"{i}/{len(syms)} · reuse {reused} · new {downloaded} · errors {len(errors)}")
    r={"study":"Historical Depth Build v1","verdict":"PASS" if not errors and len(ready)==len(syms) else "PARTIAL_FAIL",
       "symbols_expected":len(syms),"symbols_ready":len(ready),"rows_total":sum(x["rows"] for x in ready),
       "reused":reused,"downloaded":downloaded,"errors":errors,
       "next_use":"Formation/regime/walk-forward daily context only. Frozen Forward and existing intraday OOS are unchanged."}
    MAN.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding="utf-8")
    return r
