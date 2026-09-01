
import json
from pathlib import Path
import pandas as pd
import numpy as np

def _num(s): return pd.to_numeric(s, errors="coerce")
def _pf(r):
    r=_num(r).fillna(0.0); gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    return gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
def _summary(df):
    if df.empty:return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"stop_rate":0.0}
    r=_num(df["pnl_r"]).fillna(0.0)
    return {"trades":int(len(df)),"win_rate":round(float((r>0).mean()),4),"profit_factor":round(_pf(r),4),
            "avg_r":round(float(r.mean()),4),"stop_rate":round(float(df["exit_reason"].astype(str).eq("STOP").mean()),4)}
def _monthly(df):
    x=df.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce"); x=x.dropna(subset=["date"]); x["month"]=x["date"].dt.to_period("M").astype(str)
    return [{"month":m,**_summary(g)} for m,g in x.groupby("month",sort=True)]
def _halves(df):
    x=df.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce"); x=x.dropna(subset=["date"]).sort_values("date")
    if x.empty:return []
    mid=x["date"].min()+(x["date"].max()-x["date"].min())/2
    return [{"half":"H1",**_summary(x[x["date"]<=mid])},{"half":"H2",**_summary(x[x["date"]>mid])}]
def _conc(df):
    x=df.copy(); x["pnl_r"]=_num(x["pnl_r"]).fillna(0.0); w=x[x["pnl_r"]>0].sort_values("pnl_r",ascending=False)
    total=float(w["pnl_r"].sum())
    if total<=0:return {"winner_count":int(len(w)),"top1_share":0.0,"top3_share":0.0,"top5_share":0.0}
    return {"winner_count":int(len(w)),"top1_share":round(float(w.head(1)["pnl_r"].sum()/total),4),
            "top3_share":round(float(w.head(3)["pnl_r"].sum()/total),4),
            "top5_share":round(float(w.head(5)["pnl_r"].sum()/total),4)}
def _boot(df,n=3000,seed=20260901):
    if df.empty:return {"avg_r_ci95":[0,0],"pf_ci95":[0,0],"prob_avg_r_gt0":0.0,"prob_pf_gt1":0.0}
    r=_num(df["pnl_r"]).fillna(0.0).to_numpy(float); rng=np.random.default_rng(seed); av=[]; pfs=[]; N=len(r)
    for _ in range(n):
        s=r[rng.integers(0,N,size=N)]; av.append(float(s.mean())); gp=float(s[s>0].sum()); gl=float(-s[s<0].sum()); pfs.append(gp/gl if gl>0 else (20.0 if gp>0 else 0.0))
    return {"avg_r_ci95":[round(float(np.quantile(av,.025)),4),round(float(np.quantile(av,.975)),4)],
            "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
            "prob_avg_r_gt0":round(float(np.mean(np.array(av)>0)),4),"prob_pf_gt1":round(float(np.mean(np.array(pfs)>1)),4)}
def run_opening_drive_audit(discovery_path,oos_path,outdir,progress_cb=None):
    d=pd.read_csv(discovery_path); o=pd.read_csv(oos_path)
    req={"symbol","date","pnl_r","exit_reason"}
    for name,df in [("discovery",d),("oos",o)]:
        miss=sorted(req-set(df.columns))
        if miss: raise ValueError(f"{name} missing {miss}")
    if progress_cb:progress_cb(15,"月度稳定性")
    dm,om=_monthly(d),_monthly(o)
    if progress_cb:progress_cb(35,"Half-sample + Winner Concentration")
    dh,oh=_halves(d),_halves(o); dc,oc=_conc(d),_conc(o)
    if progress_cb:progress_cb(60,"Bootstrap")
    db,ob=_boot(d),_boot(o); ds,os=_summary(d),_summary(o)
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
    result={"study":"Opening Drive Audit v1","discovery":ds,"oos":os,"discovery_months":dm,"oos_months":om,
            "discovery_halves":dh,"oos_halves":oh,"discovery_winner_concentration":dc,"oos_winner_concentration":oc,
            "discovery_bootstrap":db,"oos_bootstrap":ob,"checks":checks,"checks_passed":passed,"checks_total":total,"verdict":verdict,
            "guardrail":"No threshold tuning in this audit."}
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    (out/"opening_drive_audit_v1.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if progress_cb:progress_cb(100,f"Opening Drive Audit 完成：{verdict}")
    return result
