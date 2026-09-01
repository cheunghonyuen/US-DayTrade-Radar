
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/daytrade/paper_forward_qc_v1")
LEDGER = ROOT / "paper_ledger.csv"
STATE = ROOT / "state.json"
FREEZE = ROOT / "freeze_manifest.json"

def _pf(r):
    r=pd.to_numeric(r,errors="coerce").dropna()
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    return gp/gl if gl>0 else (999.0 if gp>0 else 0.0)

def _metrics(df):
    if df.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,
                "median_r":0.0,"max_dd_r":0.0,"net_r":0.0}
    r=pd.to_numeric(df["pnl_r"],errors="coerce").dropna()
    if r.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,
                "median_r":0.0,"max_dd_r":0.0,"net_r":0.0}
    eq=r.cumsum()
    dd=eq-eq.cummax()
    return {
        "trades":int(len(r)),
        "win_rate":round(float((r>0).mean()),4),
        "profit_factor":round(float(_pf(r)),4),
        "avg_r":round(float(r.mean()),4),
        "median_r":round(float(r.median()),4),
        "max_dd_r":round(float(dd.min()),4),
        "net_r":round(float(r.sum()),4),
    }

def load_command_center():
    if not FREEZE.exists():
        raise FileNotFoundError("26 freeze_manifest.json missing. Run Frozen Paper Forward first.")
    freeze=json.loads(FREEZE.read_text(encoding="utf-8"))
    state=json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    ledger=pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()

    if ledger.empty:
        return {
            "study":"Paper Trading Command Center v1",
            "frozen":True,
            "freeze_hash":freeze.get("spec_sha256",""),
            "processed_dates":state.get("processed_dates",[]),
            "overall":_metrics(ledger),
            "daily":[], "latest_trades":[], "equity_curve":[],
            "gate":{"days":0,"trades":0,"days_target":20,"trades_target":100,
                    "days_pct":0.0,"trades_pct":0.0,"ready_for_judgement":False},
            "message":"Frozen ledger exists but has no trades yet."
        }

    ledger["pnl_r"]=pd.to_numeric(ledger["pnl_r"],errors="coerce")
    ledger=ledger.dropna(subset=["pnl_r"]).copy()
    ledger["date"]=ledger["date"].astype(str)
    sort_cols=[c for c in ["date","entry_time","symbol"] if c in ledger.columns]
    ledger=ledger.sort_values(sort_cols).reset_index(drop=True)
    ledger["cum_r"]=ledger["pnl_r"].cumsum()

    daily=[]
    for d,g in ledger.groupby("date",sort=True):
        m=_metrics(g)
        daily.append({"date":str(d),**m})

    latest_date=max(ledger["date"]) if len(ledger) else None
    latest=ledger[ledger["date"].eq(latest_date)].copy() if latest_date else ledger.iloc[0:0]
    keep=["date","symbol","side","rvol","atr14","intended_entry","stop",
          "entry_fill","entry_time","exit_fill","exit_time","exit_reason","pnl_r"]
    latest_trades=latest[[c for c in keep if c in latest.columns]].to_dict("records")

    curve=[{"n":int(i+1),"date":str(r["date"]),"symbol":str(r.get("symbol","")),
            "cum_r":round(float(r["cum_r"]),4)}
           for i,r in ledger.iterrows()]

    days=len(set(state.get("processed_dates",[])))
    trades=len(ledger)
    gate={
        "days":int(days),"trades":int(trades),
        "days_target":20,"trades_target":100,
        "days_pct":round(min(days/20,1.0),4),
        "trades_pct":round(min(trades/100,1.0),4),
        "ready_for_judgement":bool(days>=20 and trades>=100)
    }

    return {
        "study":"Paper Trading Command Center v1",
        "frozen":True,
        "freeze_hash":freeze.get("spec_sha256",""),
        "forward_start_date":freeze.get("spec",{}).get("forward_start_date",""),
        "processed_dates":state.get("processed_dates",[]),
        "latest_date":latest_date,
        "overall":_metrics(ledger),
        "daily":daily[-20:],
        "latest_trades":latest_trades,
        "equity_curve":curve,
        "gate":gate,
        "guardrail":"READ-ONLY analytics over the frozen 26 ledger. No strategy parameter is changed."
    }
