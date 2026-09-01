
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

ROOT = Path("/data/daytrade")
DAILY_DIR = ROOT / "formation_daily_v1"
DAILY_MANIFEST = DAILY_DIR / "manifest.json"
OUTDIR = ROOT / "pairs_walkforward_v1"

SPEC = {
    "family": "Pairs / statistical arbitrage",
    "methodology_basis": "252d formation + 63d trading walk-forward; pair selection only on formation; lagged positions; transaction costs",
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
    "pair_weighting": "equal-weight selected pairs",
    "lookahead": "forbidden",
    "parameter_tuning": "NONE"
}

def _load_prices():
    if not DAILY_MANIFEST.exists():
        raise FileNotFoundError("34 manifest missing; run Historical Depth Build first.")
    man=json.loads(DAILY_MANIFEST.read_text(encoding="utf-8"))
    if man.get("verdict")!="PASS":
        raise RuntimeError(f"34 build not PASS: {man.get('verdict')}")
    files=sorted(DAILY_DIR.glob("*.csv"))
    if len(files)<100:
        raise RuntimeError(f"formation cache incomplete: only {len(files)} symbol files")
    frames=[]
    for p in files:
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
    return px

def _ols_spread(y,x):
    xv=np.asarray(x,float); yv=np.asarray(y,float)
    X=np.column_stack([np.ones(len(xv)),xv])
    coef=np.linalg.lstsq(X,yv,rcond=None)[0]
    alpha,beta=float(coef[0]),float(coef[1])
    spread=yv-(alpha+beta*xv)
    return alpha,beta,spread

def _candidate_pairs(form):
    # log prices; require common history
    logp=np.log(form.replace(0,np.nan))
    valid_cols=[c for c in logp.columns if logp[c].notna().sum()>=SPEC["min_common_days"]]
    logp=logp[valid_cols]
    rets=logp.diff()
    corr=rets.corr(min_periods=SPEC["min_common_days"]-5)
    pairs=[]
    cols=list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            v=corr.iloc[i,j]
            if pd.notna(v):
                pairs.append((abs(float(v)),cols[i],cols[j]))
    pairs=sorted(pairs,reverse=True)[:SPEC["corr_prefilter_top_pairs"]]
    return [(a,b,c) for a,b,c in pairs]

def _screen_fold(form):
    picked=[]
    for abs_corr,a,b in _candidate_pairs(form):
        q=form[[a,b]].dropna()
        if len(q)<SPEC["min_common_days"]:
            continue
        y=np.log(q[a].astype(float)); x=np.log(q[b].astype(float))
        try:
            stat,pv,_=coint(y,x,trend="c",autolag="aic")
        except Exception:
            continue
        if not np.isfinite(pv) or pv>SPEC["cointegration_pvalue_max"]:
            continue
        alpha,beta,spread=_ols_spread(y.values,x.values)
        sd=float(np.std(spread,ddof=1))
        mu=float(np.mean(spread))
        if not np.isfinite(sd) or sd<=1e-8 or not np.isfinite(beta):
            continue
        picked.append({
            "a":a,"b":b,"corr":abs_corr,"pvalue":float(pv),
            "alpha":alpha,"beta":beta,"mu":mu,"sd":sd
        })
    picked=sorted(picked,key=lambda r:(r["pvalue"],-r["corr"]))
    return picked[:SPEC["max_pairs_per_fold"]]

def _trade_pair(rec,test):
    a,b=rec["a"],rec["b"]
    q=test[[a,b]].dropna().copy()
    if len(q)<5:
        return pd.DataFrame(), pd.Series(dtype=float)
    la=np.log(q[a].astype(float)); lb=np.log(q[b].astype(float))
    spread=la-(rec["alpha"]+rec["beta"]*lb)
    z=(spread-rec["mu"])/rec["sd"]

    pos=[]
    state=0
    entry_i=None
    trades=[]
    for i,(dt,zz) in enumerate(z.items()):
        if not np.isfinite(zz):
            pos.append(state); continue
        if state==0:
            if zz>=SPEC["entry_z"]:
                state=-1; entry_i=i
            elif zz<=-SPEC["entry_z"]:
                state=1; entry_i=i
        elif state==1:
            if zz>=-SPEC["exit_z"] or zz<=-SPEC["stop_z"]:
                reason="EXIT" if zz>=-SPEC["exit_z"] else "STOP"
                trades.append((entry_i,i,state,reason))
                state=0; entry_i=None
        elif state==-1:
            if zz<=SPEC["exit_z"] or zz>=SPEC["stop_z"]:
                reason="EXIT" if zz<=SPEC["exit_z"] else "STOP"
                trades.append((entry_i,i,state,reason))
                state=0; entry_i=None
        pos.append(state)

    if state!=0 and entry_i is not None:
        trades.append((entry_i,len(q)-1,state,"FOLD_END"))

    # dollar-neutral-ish pair return: long spread = long A, short beta*B.
    ra=q[a].pct_change().fillna(0.0)
    rb=q[b].pct_change().fillna(0.0)
    gross=1.0+abs(rec["beta"])
    spread_ret=(ra-rec["beta"]*rb)/gross

    # position is signal state for day t; realized next-day returns use shift(1)
    pser=pd.Series(pos,index=q.index,dtype=float).shift(1).fillna(0.0)
    daily=pser*spread_ret

    # cost charged on absolute position change; roundtrip 5 bps total -> 2.5 bps per side change
    turnover=pser.diff().abs().fillna(abs(pser.iloc[0]))
    daily -= turnover*(SPEC["cost_bps_roundtrip"]/2.0)/10000.0

    out=[]
    for ei,xi,side,reason in trades:
        if ei is None or xi<=ei: continue
        idx=q.index
        r=float((1+daily.iloc[ei+1:xi+1]).prod()-1) if xi>=ei+1 else 0.0
        out.append({
            "pair":f"{a}/{b}","a":a,"b":b,
            "entry_date":str(idx[ei].date()),"exit_date":str(idx[xi].date()),
            "side":"LONG_SPREAD" if side==1 else "SHORT_SPREAD",
            "reason":reason,"return_pct":r*100,
            "pvalue":rec["pvalue"],"beta":rec["beta"]
        })
    return pd.DataFrame(out),daily

def _metrics(trades,daily):
    if len(daily)==0:
        return {"trades":0,"win_rate":0,"profit_factor":0,"avg_trade_pct":0,"sharpe":0,"max_dd_pct":0,"net_return_pct":0}
    rr=pd.to_numeric(trades["return_pct"],errors="coerce").dropna()/100 if not trades.empty else pd.Series(dtype=float)
    gp=float(rr[rr>0].sum()); gl=float(-rr[rr<0].sum())
    pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    d=pd.to_numeric(daily,errors="coerce").fillna(0)
    eq=(1+d).cumprod()
    dd=eq/eq.cummax()-1
    sd=float(d.std(ddof=1))
    sharpe=float(d.mean()/sd*np.sqrt(252)) if sd>0 else 0.0
    return {
        "trades":int(len(rr)),
        "win_rate":round(float((rr>0).mean()),4) if len(rr) else 0.0,
        "profit_factor":round(float(pf),4),
        "avg_trade_pct":round(float(rr.mean()*100),4) if len(rr) else 0.0,
        "sharpe":round(sharpe,4),
        "max_dd_pct":round(float(dd.min()*100),4),
        "net_return_pct":round(float((eq.iloc[-1]-1)*100),4)
    }

def _bootstrap(trades,n=2000,seed=20260901):
    if trades.empty:
        return {"prob_pf_gt1":0.0,"prob_avg_trade_gt0":0.0}
    r=pd.to_numeric(trades["return_pct"],errors="coerce").dropna().to_numpy(float)
    if len(r)==0:return {"prob_pf_gt1":0.0,"prob_avg_trade_gt0":0.0}
    rng=np.random.default_rng(seed); pfs=[]; av=[]
    for _ in range(n):
        s=r[rng.integers(0,len(r),len(r))]
        gp=s[s>0].sum(); gl=-s[s<0].sum()
        pfs.append(gp/gl if gl>0 else (20 if gp>0 else 0))
        av.append(s.mean())
    return {
        "prob_pf_gt1":round(float((np.array(pfs)>1).mean()),4),
        "prob_avg_trade_gt0":round(float((np.array(av)>0).mean()),4),
        "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
        "avg_trade_ci95_pct":[round(float(np.quantile(av,.025)),4),round(float(np.quantile(av,.975)),4)]
    }

def run_pairs_walkforward(progress_cb=None):
    px=_load_prices()
    dates=px.index.unique().sort_values()
    F=SPEC["formation_days"]; T=SPEC["test_days"]; S=SPEC["step_days"]
    if len(dates)<F+T:
        raise RuntimeError(f"insufficient daily depth: {len(dates)} rows")

    OUTDIR.mkdir(parents=True,exist_ok=True)
    folds=[]
    all_trades=[]
    daily_parts=[]
    starts=list(range(F,len(dates)-T+1,S))
    total=max(len(starts),1)

    if progress_cb:
        progress_cb(2,f"Preflight PASS · {px.shape[1]} symbols · {len(dates)} dates · {len(starts)} folds")

    for fi,start in enumerate(starts,1):
        form_dates=dates[start-F:start]
        test_dates=dates[start:start+T]
        form=px.loc[form_dates]
        test=px.loc[test_dates]
        selected=_screen_fold(form)

        fold_trades=[]
        pair_daily=[]
        for rec in selected:
            tr,dy=_trade_pair(rec,test)
            if not tr.empty: fold_trades.append(tr)
            if len(dy): pair_daily.append(dy.rename(f"{rec['a']}/{rec['b']}"))

        ftr=pd.concat(fold_trades,ignore_index=True) if fold_trades else pd.DataFrame()
        if pair_daily:
            dd=pd.concat(pair_daily,axis=1).fillna(0).mean(axis=1)
        else:
            dd=pd.Series(0.0,index=test_dates)

        fm=_metrics(ftr,dd)
        folds.append({
            "fold":fi,
            "formation_start":str(form_dates[0].date()),
            "formation_end":str(form_dates[-1].date()),
            "test_start":str(test_dates[0].date()),
            "test_end":str(test_dates[-1].date()),
            "pairs_selected":len(selected),
            **fm
        })
        if not ftr.empty:
            ftr["fold"]=fi
            all_trades.append(ftr)
        daily_parts.append(dd)

        if progress_cb:
            progress_cb(5+int(85*fi/total),f"Fold {fi}/{total} · pairs {len(selected)} · trades {len(ftr)} · PF {fm['profit_factor']}")

    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    daily=pd.concat(daily_parts).groupby(level=0).mean().sort_index() if daily_parts else pd.Series(dtype=float)
    overall=_metrics(trades,daily)
    bs=_bootstrap(trades)
    fdf=pd.DataFrame(folds)

    trades.to_csv(OUTDIR/"trades.csv",index=False)
    fdf.to_csv(OUTDIR/"folds.csv",index=False)
    pd.DataFrame({"date":daily.index.astype(str),"return":daily.values}).to_csv(OUTDIR/"daily_returns.csv",index=False)

    positive_folds=int((pd.to_numeric(fdf["profit_factor"],errors="coerce")>1).sum()) if not fdf.empty else 0
    result={
        "study":"Walk-Forward Pairs / Statistical Arbitrage v1",
        "spec":SPEC,
        "data":{"symbols":int(px.shape[1]),"dates":int(len(dates)),"date_min":str(dates.min().date()),"date_max":str(dates.max().date())},
        "folds":folds,
        "overall":overall,
        "bootstrap":bs,
        "positive_pf_folds":positive_folds,
        "fold_count":int(len(folds)),
        "method_note":"Pair selection and cointegration are formation-only. Correlation is only a prefilter to cap pair tests because sector metadata is not yet present. Test-window PnL uses lagged positions and 5 bps round-trip cost.",
        "decision_rule":"PROMOTE only if overall PF>1, avg trade>0, Sharpe>0, bootstrap P(PF>1)>=80%, and at least 60% of folds have PF>1. Otherwise DROP or keep as research-only; no rescue tuning."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Pairs WF complete · {len(folds)} folds · {overall['trades']} trades · PF {overall['profit_factor']} · Sharpe {overall['sharpe']}")
    return result
