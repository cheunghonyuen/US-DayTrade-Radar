
import gc, json, math, os, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

ROOT = Path("/data/daytrade")
DAILY_DIR = ROOT / "formation_daily_v1"
DAILY_MANIFEST = DAILY_DIR / "manifest.json"
OUTDIR = ROOT / "pairs_walkforward_v2"
STATE = OUTDIR / "state.json"
FOLDS_DIR = OUTDIR / "folds"
REQ_COLS = {"date","close"}

SPEC = {
    "formation_days": 252,
    "test_days": 63,
    "step_days": 63,
    "min_common_days": 220,
    "corr_prefilter_top_pairs": 150,
    "cointegration_pvalue_max": 0.10,
    "max_pairs_per_fold": 10,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "stop_z": 3.0,
    "cost_bps_roundtrip": 5.0,
    "checkpoint_every_pair_tests": 15,
    "max_pair_tests_per_batch": 30,
    "parameter_tuning": "NONE"
}

def _rss_mb():
    try:
        # Linux ru_maxrss is KB.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)
    except Exception:
        return None

def _load_manifest():
    if not DAILY_MANIFEST.exists():
        raise FileNotFoundError("34 manifest missing")
    m=json.loads(DAILY_MANIFEST.read_text(encoding="utf-8"))
    if m.get("verdict")!="PASS":
        raise RuntimeError(f"34 build not PASS: {m.get('verdict')}")
    return m

def _preflight_files():
    m=_load_manifest()
    files=sorted(DAILY_DIR.glob("*.csv"))
    issues=[]
    rows=0
    sample=[]
    for i,p in enumerate(files):
        try:
            q=pd.read_csv(p,nrows=5)
            miss=sorted(REQ_COLS-set(q.columns))
            if miss: issues.append({"file":p.name,"error":f"missing {miss}"})
            if i<5: sample.append({"file":p.name,"columns":list(q.columns)})
        except Exception as e:
            issues.append({"file":p.name,"error":f"{type(e).__name__}: {e}"})
    return {
        "manifest_verdict":m.get("verdict"),
        "files":len(files),
        "expected":m.get("symbols_expected"),
        "issues":issues[:20],
        "sample":sample,
        "rss_mb":_rss_mb(),
        "statsmodels_coint_available":True
    }

def _load_prices():
    frames=[]
    for p in sorted(DAILY_DIR.glob("*.csv")):
        sym=p.stem
        x=pd.read_csv(p,usecols=["date","close"])
        x["date"]=pd.to_datetime(x["date"],errors="coerce")
        x["close"]=pd.to_numeric(x["close"],errors="coerce")
        x=x.dropna().drop_duplicates("date").set_index("date").sort_index()
        if len(x)>=100:
            frames.append(x.rename(columns={"close":sym}))
    if not frames:
        raise RuntimeError("no valid daily files")
    px=pd.concat(frames,axis=1).sort_index()
    del frames
    gc.collect()
    return px

def _ols_spread(y,x):
    xv=np.asarray(x,float); yv=np.asarray(y,float)
    X=np.column_stack([np.ones(len(xv)),xv])
    coef=np.linalg.lstsq(X,yv,rcond=None)[0]
    a,b=float(coef[0]),float(coef[1])
    spread=yv-(a+b*xv)
    return a,b,spread

def _candidate_pairs(form):
    logp=np.log(form.replace(0,np.nan))
    cols=[c for c in logp.columns if logp[c].notna().sum()>=SPEC["min_common_days"]]
    logp=logp[cols]
    rets=logp.diff()
    corr=rets.corr(min_periods=SPEC["min_common_days"]-5)
    pairs=[]
    cols=list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            v=corr.iat[i,j]
            if pd.notna(v):
                pairs.append((abs(float(v)),cols[i],cols[j]))
    pairs=sorted(pairs,reverse=True)[:SPEC["corr_prefilter_top_pairs"]]
    del logp,rets,corr
    gc.collect()
    return pairs

def _trade_pair(rec,test):
    a,b=rec["a"],rec["b"]
    q=test[[a,b]].dropna().copy()
    if len(q)<5:
        return pd.DataFrame(), pd.Series(dtype=float)
    la=np.log(q[a].astype(float)); lb=np.log(q[b].astype(float))
    spread=la-(rec["alpha"]+rec["beta"]*lb)
    z=(spread-rec["mu"])/rec["sd"]

    states=[]; state=0; entry_i=None; spans=[]
    for i,zz in enumerate(z.to_numpy(float)):
        if not np.isfinite(zz):
            states.append(state); continue
        if state==0:
            if zz>=SPEC["entry_z"]: state=-1; entry_i=i
            elif zz<=-SPEC["entry_z"]: state=1; entry_i=i
        elif state==1 and (zz>=-SPEC["exit_z"] or zz<=-SPEC["stop_z"]):
            spans.append((entry_i,i,state,"EXIT" if zz>=-SPEC["exit_z"] else "STOP")); state=0; entry_i=None
        elif state==-1 and (zz<=SPEC["exit_z"] or zz>=SPEC["stop_z"]):
            spans.append((entry_i,i,state,"EXIT" if zz<=SPEC["exit_z"] else "STOP")); state=0; entry_i=None
        states.append(state)
    if state!=0 and entry_i is not None:
        spans.append((entry_i,len(q)-1,state,"FOLD_END"))

    ra=q[a].pct_change().fillna(0.0); rb=q[b].pct_change().fillna(0.0)
    gross=1+abs(rec["beta"])
    pairret=(ra-rec["beta"]*rb)/gross
    pser=pd.Series(states,index=q.index,dtype=float).shift(1).fillna(0.0)
    daily=pser*pairret
    turnover=pser.diff().abs().fillna(abs(pser.iloc[0]))
    daily-=turnover*(SPEC["cost_bps_roundtrip"]/2)/10000

    rows=[]
    for ei,xi,side,reason in spans:
        if ei is None or xi<=ei: continue
        r=float((1+daily.iloc[ei+1:xi+1]).prod()-1)
        rows.append({
            "pair":f"{a}/{b}","a":a,"b":b,
            "entry_date":str(q.index[ei].date()),"exit_date":str(q.index[xi].date()),
            "side":"LONG_SPREAD" if side==1 else "SHORT_SPREAD",
            "reason":reason,"return_pct":r*100,"pvalue":rec["pvalue"],"beta":rec["beta"]
        })
    return pd.DataFrame(rows),daily

def _metrics(trades,daily):
    rr=pd.to_numeric(trades["return_pct"],errors="coerce").dropna()/100 if not trades.empty else pd.Series(dtype=float)
    gp=float(rr[rr>0].sum()); gl=float(-rr[rr<0].sum())
    pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    d=pd.to_numeric(daily,errors="coerce").fillna(0) if len(daily) else pd.Series(dtype=float)
    if len(d):
        eq=(1+d).cumprod()
        dd=eq/eq.cummax()-1
        sd=float(d.std(ddof=1))
        sharpe=float(d.mean()/sd*np.sqrt(252)) if sd>0 else 0.0
        maxdd=float(dd.min()*100); net=float((eq.iloc[-1]-1)*100)
    else:
        sharpe=maxdd=net=0.0
    return {"trades":int(len(rr)),"win_rate":round(float((rr>0).mean()),4) if len(rr) else 0,
            "profit_factor":round(float(pf),4),"avg_trade_pct":round(float(rr.mean()*100),4) if len(rr) else 0,
            "sharpe":round(sharpe,4),"max_dd_pct":round(maxdd,4),"net_return_pct":round(net,4)}

def _load_state():
    if STATE.exists():
        try:return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:pass
    return {"completed_folds":[]}

def _save_state(st):
    OUTDIR.mkdir(parents=True,exist_ok=True); FOLDS_DIR.mkdir(parents=True,exist_ok=True)
    tmp=STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st,indent=2,ensure_ascii=False),encoding="utf-8"); tmp.replace(STATE)

def _run_one_fold(px,fi,start,dates,progress_cb=None):
    F,T=SPEC["formation_days"],SPEC["test_days"]
    fdates=dates[start-F:start]; tdates=dates[start:start+T]
    form=px.loc[fdates].copy(); test=px.loc[tdates].copy()

    candidates=_candidate_pairs(form)
    selected=[]
    tested=0

    for batch0 in range(0,len(candidates),SPEC["max_pair_tests_per_batch"]):
        batch=candidates[batch0:batch0+SPEC["max_pair_tests_per_batch"]]
        for abs_corr,a,b in batch:
            q=form[[a,b]].dropna()
            if len(q)<SPEC["min_common_days"]: continue
            y=np.log(q[a].astype(float)); x=np.log(q[b].astype(float))
            try:
                _,pv,_=coint(y,x,trend="c",autolag="aic")
            except Exception:
                continue
            tested+=1
            if np.isfinite(pv) and pv<=SPEC["cointegration_pvalue_max"]:
                alpha,beta,spread=_ols_spread(y.values,x.values)
                sd=float(np.std(spread,ddof=1)); mu=float(np.mean(spread))
                if np.isfinite(sd) and sd>1e-8 and np.isfinite(beta):
                    selected.append({"a":a,"b":b,"corr":abs_corr,"pvalue":float(pv),
                                     "alpha":alpha,"beta":beta,"mu":mu,"sd":sd})
        selected=sorted(selected,key=lambda r:(r["pvalue"],-r["corr"]))[:SPEC["max_pairs_per_fold"]]
        gc.collect()
        if progress_cb:
            progress_cb(None,f"Fold {fi}: pair tests {min(batch0+len(batch),len(candidates))}/{len(candidates)} · selected {len(selected)} · RSS {_rss_mb()} MB")

    fold_trades=[]; dailies=[]
    for rec in selected:
        tr,dy=_trade_pair(rec,test)
        if not tr.empty: fold_trades.append(tr)
        if len(dy): dailies.append(dy.rename(f"{rec['a']}/{rec['b']}"))
    ftr=pd.concat(fold_trades,ignore_index=True) if fold_trades else pd.DataFrame()
    dd=pd.concat(dailies,axis=1).fillna(0).mean(axis=1) if dailies else pd.Series(0.0,index=tdates)
    met=_metrics(ftr,dd)

    fd=FOLDS_DIR/f"fold_{fi:02d}"
    fd.mkdir(parents=True,exist_ok=True)
    ftr.to_csv(fd/"trades.csv",index=False)
    pd.DataFrame({"date":dd.index.astype(str),"return":dd.values}).to_csv(fd/"daily_returns.csv",index=False)
    meta={"fold":fi,"formation_start":str(fdates[0].date()),"formation_end":str(fdates[-1].date()),
          "test_start":str(tdates[0].date()),"test_end":str(tdates[-1].date()),
          "pair_tests":tested,"pairs_selected":len(selected),"selected_pairs":selected,**met}
    (fd/"meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")

    del form,test,candidates,selected,fold_trades,dailies,ftr,dd
    gc.collect()
    return meta

def _aggregate():
    metas=[]; trs=[]; dys=[]
    for fd in sorted(FOLDS_DIR.glob("fold_*")):
        mp=fd/"meta.json"
        if not mp.exists(): continue
        metas.append(json.loads(mp.read_text(encoding="utf-8")))
        tp=fd/"trades.csv"
        if tp.exists():
            t=pd.read_csv(tp)
            if not t.empty: trs.append(t)
        dp=fd/"daily_returns.csv"
        if dp.exists():
            d=pd.read_csv(dp)
            if not d.empty:
                s=pd.Series(pd.to_numeric(d["return"],errors="coerce").fillna(0).values,index=pd.to_datetime(d["date"]))
                dys.append(s)
    trades=pd.concat(trs,ignore_index=True) if trs else pd.DataFrame()
    daily=pd.concat(dys).groupby(level=0).mean().sort_index() if dys else pd.Series(dtype=float)
    overall=_metrics(trades,daily)
    return metas,trades,daily,overall

def _bootstrap(trades,n=2000,seed=20260901):
    if trades.empty:return {"prob_pf_gt1":0.0,"prob_avg_trade_gt0":0.0}
    r=pd.to_numeric(trades["return_pct"],errors="coerce").dropna().to_numpy(float)
    if len(r)==0:return {"prob_pf_gt1":0.0,"prob_avg_trade_gt0":0.0}
    rng=np.random.default_rng(seed); pfs=[]; av=[]
    for _ in range(n):
        s=r[rng.integers(0,len(r),len(r))]
        gp=s[s>0].sum(); gl=-s[s<0].sum()
        pfs.append(gp/gl if gl>0 else (20 if gp>0 else 0)); av.append(s.mean())
    return {"prob_pf_gt1":round(float((np.array(pfs)>1).mean()),4),
            "prob_avg_trade_gt0":round(float((np.array(av)>0).mean()),4),
            "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)]}

def run_pairs_runtime_preflight(progress_cb=None):
    pf=_preflight_files()
    if pf["files"]!=pf["expected"] or pf["issues"]:
        verdict="FAIL"
    else:
        verdict="PASS"
    r={"study":"Pairs Runtime/Schema Preflight v2","verdict":verdict,**pf,
       "guardrail":"Long run blocked unless PASS. V2 checkpoints every fold, batch-tests pair coint, frees DataFrames, and resumes completed folds."}
    OUTDIR.mkdir(parents=True,exist_ok=True)
    (OUTDIR/"preflight.json").write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb: progress_cb(100,f"Preflight {verdict} · {pf['files']} files · RSS {pf['rss_mb']} MB")
    return r

def run_pairs_walkforward_resumable(progress_cb=None):
    pre=OUTDIR/"preflight.json"
    if not pre.exists(): raise RuntimeError("Run v2 Preflight first")
    p=json.loads(pre.read_text(encoding="utf-8"))
    if p.get("verdict")!="PASS": raise RuntimeError("Preflight not PASS")

    px=_load_prices()
    dates=px.index.unique().sort_values()
    F,T,S=SPEC["formation_days"],SPEC["test_days"],SPEC["step_days"]
    starts=list(range(F,len(dates)-T+1,S))
    st=_load_state(); done=set(int(x) for x in st.get("completed_folds",[]))
    total=len(starts)

    if progress_cb:
        progress_cb(2,f"Loaded {px.shape[1]} symbols × {len(dates)} dates · {len(done)}/{total} folds already complete · RSS {_rss_mb()} MB")

    for idx,start in enumerate(starts,1):
        if idx in done:
            continue
        def cb(_,msg):
            if progress_cb:
                base=5+int(85*(idx-1)/max(total,1))
                progress_cb(base,msg)
        meta=_run_one_fold(px,idx,start,dates,cb)
        done.add(idx)
        st["completed_folds"]=sorted(done)
        st["last_fold"]=idx
        st["last_meta"]=meta
        _save_state(st)
        gc.collect()
        if progress_cb:
            progress_cb(5+int(85*idx/max(total,1)),f"Checkpoint fold {idx}/{total} · PF {meta['profit_factor']} · trades {meta['trades']} · RSS {_rss_mb()} MB")

    metas,trades,daily,overall=_aggregate()
    bs=_bootstrap(trades)
    pos=int(sum(1 for m in metas if float(m.get("profit_factor",0))>1))
    result={"study":"Walk-Forward Pairs / Statistical Arbitrage v2","spec":SPEC,
            "fold_count":len(metas),"completed_folds":len(done),"positive_pf_folds":pos,
            "overall":overall,"bootstrap":bs,"folds":metas,
            "runtime":{"rss_mb":_rss_mb(),"resumable":True,"checkpoint_dir":str(FOLDS_DIR)},
            "decision_rule":"PROMOTE only if PF>1, avg trade>0, Sharpe>0, bootstrap P(PF>1)>=80%, and >=60% folds PF>1. No rescue tuning."}
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb: progress_cb(100,f"Complete · {len(metas)} folds · {overall['trades']} trades · PF {overall['profit_factor']} · Sharpe {overall['sharpe']}")
    return result
