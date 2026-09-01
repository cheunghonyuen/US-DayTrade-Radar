
import json
from pathlib import Path
import numpy as np
import pandas as pd
from engine import prepare

# Frozen BEFORE seeing results. No threshold sweep.
RULE = {
    "min_gap_pct": 0.02,
    "opening_window_minutes": 30,
    "min_opening_return_pct": 0.02,
    "min_opening_median_rvol": 2.5,
    "min_close_location": 0.75,
    "entry_start": "10:00",
    "entry_end": "11:30",
    "target_r": 2.0,
    "stop_range_fraction": 0.5,
}

def _n(s): return pd.to_numeric(s, errors="coerce")

def _summary(df):
    if df.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"max_dd_r":0.0,"stop_rate":0.0}
    r=_n(df["pnl_r"]).fillna(0.0)
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    eq=r.cumsum(); peak=eq.cummax(); dd=eq-peak
    return {
        "trades":int(len(df)),
        "win_rate":round(float((r>0).mean()),4),
        "profit_factor":round(float(pf),4),
        "avg_r":round(float(r.mean()),4),
        "max_dd_r":round(float(dd.min()),4),
        "stop_rate":round(float(df["exit_reason"].astype(str).eq("STOP").mean()),4),
    }

def _halves(df):
    x=df.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x=x.dropna(subset=["date"]).sort_values("date")
    if x.empty:return []
    mid=x["date"].min()+(x["date"].max()-x["date"].min())/2
    return [{"half":"H1",**_summary(x[x["date"]<=mid])},{"half":"H2",**_summary(x[x["date"]>mid])}]

def _conc(df):
    r=_n(df["pnl_r"]).fillna(0.0); w=r[r>0].sort_values(ascending=False); total=float(w.sum())
    if total<=0:return {"winner_count":int(len(w)),"top1_share":0.0,"top3_share":0.0,"top5_share":0.0}
    return {
        "winner_count":int(len(w)),
        "top1_share":round(float(w.head(1).sum()/total),4),
        "top3_share":round(float(w.head(3).sum()/total),4),
        "top5_share":round(float(w.head(5).sum()/total),4),
    }

def _boot(df,n=3000,seed=20260901):
    r=_n(df["pnl_r"]).fillna(0.0).to_numpy(float)
    if len(r)==0:return {"avg_r_ci95":[0,0],"pf_ci95":[0,0],"prob_avg_r_gt0":0.0,"prob_pf_gt1":0.0}
    rng=np.random.default_rng(seed); av=[]; pfs=[]; N=len(r)
    for _ in range(n):
        s=r[rng.integers(0,N,N)]
        av.append(float(s.mean()))
        gp=float(s[s>0].sum()); gl=float(-s[s<0].sum())
        pfs.append(gp/gl if gl>0 else (20.0 if gp>0 else 0.0))
    return {
        "avg_r_ci95":[round(float(np.quantile(av,.025)),4),round(float(np.quantile(av,.975)),4)],
        "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
        "prob_avg_r_gt0":round(float(np.mean(np.array(av)>0)),4),
        "prob_pf_gt1":round(float(np.mean(np.array(pfs)>1)),4),
    }

def _prepare_sessions(path):
    raw=pd.read_csv(path)
    x=prepare(raw).copy()
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
    x=x.dropna(subset=["timestamp"]).copy()
    x["date"]=x["timestamp"].dt.strftime("%Y-%m-%d")
    x=x[(x["time"]>="09:30")&(x["time"]<"16:00")].copy()
    x.sort_values(["symbol","timestamp"],inplace=True)

    groups={}
    for (sym,dt),g in x.groupby(["symbol","date"],sort=True):
        g=g.copy().reset_index(drop=True)
        typ=(g["high"]+g["low"]+g["close"])/3.0
        den=g["volume"].cumsum().replace(0,np.nan)
        g["vwap_mc"]=(typ*g["volume"]).cumsum()/den
        groups[(str(sym),str(dt))]=g

    last=x.groupby(["symbol","date"],sort=True)["close"].last().reset_index(name="session_close")
    last["prev_close"]=last.groupby("symbol")["session_close"].shift(1)
    prev={(str(r.symbol),str(r.date)):float(r.prev_close) for r in last.itertuples(index=False)
          if pd.notna(r.prev_close) and float(r.prev_close)>0}
    return groups,prev

def _exit(reg,entry_ts,entry,stop,target,cfg):
    post=reg[reg["timestamp"]>=entry_ts]
    if post.empty:return None
    for b in post.itertuples(index=False):
        if float(b.low)<=stop:return stop-cfg["slippage_per_share"],b.timestamp,"STOP"
        if float(b.high)>=target:return target-cfg["slippage_per_share"],b.timestamp,"TARGET"
        if b.time>=cfg["eod_exit"]:return float(b.close)-cfg["slippage_per_share"],b.timestamp,"EOD"
    z=post.iloc[-1]
    return float(z["close"])-cfg["slippage_per_share"],z["timestamp"],"LAST_BAR"

def _signal(reg,prev,cfg):
    open_px=float(reg.iloc[0]["open"])
    gap=open_px/prev-1.0
    if gap < RULE["min_gap_pct"]:
        return None

    opening=reg[reg["time"]<"10:00"]
    if len(opening)<4:
        return None

    first_close=float(opening.iloc[-1]["close"])
    opening_ret=first_close/open_px-1.0
    if opening_ret < RULE["min_opening_return_pct"]:
        return None

    med_rvol=float(pd.to_numeric(opening["rvol_bar"],errors="coerce").median())
    if not np.isfinite(med_rvol) or med_rvol < RULE["min_opening_median_rvol"]:
        return None

    hi=float(opening["high"].max()); lo=float(opening["low"].min())
    rng=max(hi-lo,0.01)
    close_location=(first_close-lo)/rng
    if close_location < RULE["min_close_location"]:
        return None

    c=reg[(reg["time"]>=RULE["entry_start"])&(reg["time"]<=RULE["entry_end"])].copy()
    c=c[(c["high"]>hi)&(c["close"]>c["vwap_mc"])]
    if c.empty:
        return None

    sig=c.iloc[0]
    later=reg[reg["timestamp"]>sig["timestamp"]]
    if later.empty:
        return None

    e=later.iloc[0].copy()
    entry=float(e["open"])+cfg["slippage_per_share"]
    stop=max(lo, entry - RULE["stop_range_fraction"]*rng)
    risk=max(entry-stop,0.01)
    target=entry+RULE["target_r"]*risk
    return e,stop,target

def _run_period(path,cfg,progress_cb=None,p0=0,p1=40,label="period"):
    groups,prev=_prepare_sessions(path)
    equity=float(cfg["starting_equity"]); rows=[]; keys=sorted(groups); total=len(keys)
    for i,key in enumerate(keys,1):
        reg=groups[key]; pv=prev.get(key)
        if pv is None or len(reg)<20: continue
        sig=_signal(reg,pv,cfg)
        if sig is None: continue
        e,stop,target=sig
        entry=float(e["open"])+cfg["slippage_per_share"]
        risk=max(entry-stop,0.01)
        shares=int((equity*cfg["risk_per_trade_pct"])//risk)
        if shares<=0: continue
        out=_exit(reg,e["timestamp"],entry,stop,target,cfg)
        if not out: continue
        xp,xt,reason=out
        pnl=(xp-entry)*shares-cfg["commission_per_share"]*shares*2
        pnl_r=pnl/(risk*shares); equity+=pnl
        rows.append({
            "setup":"MOMENTUM_CONTINUATION","symbol":key[0],"date":key[1],
            "entry_time":str(e["timestamp"]),"exit_time":str(xt),
            "pnl":round(pnl,2),"pnl_r":round(pnl_r,4),"exit_reason":reason,
            "gap_pct":round(float(reg.iloc[0]["open"]/pv-1.0),5)
        })
        if progress_cb and (i%1000==0 or i==total):
            progress_cb(p0+int((p1-p0)*i/max(total,1)),f"{label} {i}/{total}")
    df=pd.DataFrame(rows)
    return df,_summary(df)

def run_momentum_continuation_final_trial(discovery_path,oos_path,config_path,outdir,progress_cb=None):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))

    if progress_cb:progress_cb(5,"Momentum Continuation：Discovery frozen run")
    d,ds=_run_period(discovery_path,cfg,progress_cb,5,38,"Discovery")

    if progress_cb:progress_cb(42,"Momentum Continuation：True OOS frozen run")
    o,os=_run_period(oos_path,cfg,progress_cb,42,72,"True OOS")

    if progress_cb:progress_cb(75,"H1/H2 + Bootstrap")
    dh,oh=_halves(d),_halves(o)
    dc,oc=_conc(d),_conc(o)
    db,ob=_boot(d),_boot(o)

    checks={
        "discovery_pf_gt1":ds["profit_factor"]>1,
        "oos_pf_gt1":os["profit_factor"]>1,
        "discovery_avg_r_gt0":ds["avg_r"]>0,
        "oos_avg_r_gt0":os["avg_r"]>0,
        "oos_trades_ge40":os["trades"]>=40,
        "oos_bootstrap_pf_prob_ge80pct":ob["prob_pf_gt1"]>=0.80,
        "oos_bootstrap_avg_r_prob_ge80pct":ob["prob_avg_r_gt0"]>=0.80,
        "oos_top5_winner_share_le70pct":oc["top5_share"]<=0.70,
        "oos_both_halves_pf_gt1":len(oh)==2 and all(x["profit_factor"]>1 for x in oh),
    }
    passed=sum(bool(v) for v in checks.values()); total=len(checks)
    verdict="AUDIT_PASS" if passed==total else ("PROMISING" if passed>=total-2 else "AUDIT_FAIL")
    decision="KEEP_FOR_PAPER" if verdict=="AUDIT_PASS" else "DROP_MOMENTUM_CONTINUATION"

    result={
        "study":"Momentum Continuation Final Trial v1",
        "frozen_rule":RULE,
        "discovery":ds,"oos":os,
        "discovery_halves":dh,"oos_halves":oh,
        "discovery_winner_concentration":dc,"oos_winner_concentration":oc,
        "discovery_bootstrap":db,"oos_bootstrap":ob,
        "checks":checks,"checks_passed":passed,"checks_total":total,
        "verdict":verdict,"decision":decision,
        "guardrail":"One fixed hypothesis only. No threshold sweep, no redownload. If FAIL, stop this DayTrade setup round."
    }

    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"momentum_continuation_discovery.csv",index=False)
    o.to_csv(out/"momentum_continuation_oos.csv",index=False)
    (out/"momentum_continuation_final_trial_v1.json").write_text(json.dumps(result,indent=2),encoding="utf-8")

    if progress_cb:progress_cb(100,f"Momentum Continuation Final Trial：{verdict} → {decision}")
    return result
