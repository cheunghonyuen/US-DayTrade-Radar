
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/daytrade")
DISCOVERY = DATA_ROOT / "sip_data.csv"
TRUE_OOS = DATA_ROOT / "oos_v08" / "sip_oos_2024-08-30_2025-08-29.csv"
OUTDIR = DATA_ROOT / "external_challenger3_vwap_mean_reversion_v1"

REQ = {"symbol","timestamp","open","high","low","close","volume"}

SPEC = {
    "source_repo": "555aaditya/Stock-Backtesting-Pipeline",
    "source_doc": "STRATEGIES.md / vwap_mean_reversion",
    "family": "VWAP mean reversion",
    "timeframe": "5-minute",
    "vwap_std_multiplier": 1.5,
    "min_volume_ratio": 1.2,
    "typical_price": "(high+low+close)/3",
    "entry_long": "close < VWAP - 1.5*session_std(close) AND volume > session_avg_volume*1.2",
    "entry_short": "close > VWAP + 1.5*session_std(close) AND volume > session_avg_volume*1.2",
    "exit_long": "close returns to/above VWAP",
    "exit_short": "close returns to/below VWAP",
    "eod_flatten": True,
    "max_one_trade_per_symbol_day": True,
    "slippage_per_share": 0.01,
    "commission_per_share": 0.005,
    "notional_per_trade": 10000.0,
    "parameter_tuning": "NONE",
    "universe_constraint": "Source was documented mainly for Indian liquid futures/stocks; this test intentionally applies the exact documented defaults to the same US 125-stock persistent pool."
}

def _manifest(path,label):
    p=Path(path)
    info={"label":label,"path":str(p),"exists":p.exists(),
          "size_mb":round(p.stat().st_size/1024/1024,2) if p.exists() else 0.0}
    if p.exists():
        q=pd.read_csv(p,nrows=5000)
        info["columns"]=list(q.columns)
        info["required_columns_ok"]=bool(REQ.issubset(set(q.columns)))
        if "timestamp" in q.columns:
            ts=pd.to_datetime(q["timestamp"],errors="coerce",utc=True)
            info["sample_timestamp_min"]=str(ts.min())
            info["sample_timestamp_max"]=str(ts.max())
    return info

def _prepare(path,label):
    x=pd.read_csv(path)
    miss=sorted(REQ-set(x.columns))
    if miss:
        raise ValueError(f"{label} missing columns: {miss}")
    if x.duplicated(["symbol","timestamp"]).any():
        raise ValueError(f"{label} duplicate symbol/timestamp")
    ts=pd.to_datetime(x["timestamp"],errors="coerce",utc=True)
    if ts.isna().any():
        raise ValueError(f"{label} invalid timestamps: {int(ts.isna().sum())}")
    x["timestamp"]=ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["symbol","timestamp","open","high","low","close","volume"]).copy()
    x["date"]=x["timestamp"].dt.date
    x["minute"]=x["timestamp"].dt.hour*60+x["timestamp"].dt.minute
    x=x[(x["minute"]>=570)&(x["minute"]<=955)].copy()
    x.sort_values(["symbol","timestamp"],inplace=True)
    return x

def _slip(price,side,is_entry):
    s=SPEC["slippage_per_share"]
    if side=="LONG":
        return float(price)+(s if is_entry else -s)
    return float(price)+(-s if is_entry else s)

def _run_period(x,label,progress_cb=None,p0=0,p1=100):
    trades=[]
    sessions=list(x.groupby(["symbol","date"],sort=False))
    total=max(len(sessions),1)

    for n,((sym,d),g0) in enumerate(sessions,1):
        g=g0.sort_values("timestamp").reset_index(drop=True).copy()
        if len(g)<6:
            continue

        tp=(g["high"].astype(float)+g["low"].astype(float)+g["close"].astype(float))/3.0
        pv=tp*g["volume"].astype(float)
        cumv=g["volume"].astype(float).cumsum()
        g["vwap"]=pv.cumsum()/cumv.replace(0,np.nan)

        # Document says "standard deviation of intraday closes"; use expanding session std
        # so only data available at each bar is used.
        g["session_std"]=g["close"].astype(float).expanding(min_periods=5).std(ddof=0)

        # "avg_volume" is not further specified in the source doc. To avoid look-ahead,
        # use expanding prior-bar mean volume; current bar is compared to PREVIOUS average.
        g["avg_volume_prior"]=g["volume"].astype(float).shift(1).expanding(min_periods=5).mean()
        g["upper"]=g["vwap"]+SPEC["vwap_std_multiplier"]*g["session_std"]
        g["lower"]=g["vwap"]-SPEC["vwap_std_multiplier"]*g["session_std"]

        entered=False
        side=None
        entry_fill=None
        entry_i=None
        entry_vwap=None

        for i in range(5,len(g)-1):
            r=g.iloc[i]
            if not entered:
                if pd.isna(r["session_std"]) or pd.isna(r["avg_volume_prior"]) or float(r["avg_volume_prior"])<=0:
                    continue
                vol_ok=float(r["volume"]) > float(r["avg_volume_prior"])*SPEC["min_volume_ratio"]
                long_sig=float(r["close"]) < float(r["lower"]) and vol_ok
                short_sig=float(r["close"]) > float(r["upper"]) and vol_ok
                if not (long_sig or short_sig):
                    continue

                side="LONG" if long_sig else "SHORT"
                # Signal is known at current 5m close; fill at next bar open.
                raw=float(g.iloc[i+1]["open"])
                entry_fill=_slip(raw,side,True)
                entry_i=i+1
                entry_vwap=float(r["vwap"])
                entered=True
                continue

            r=g.iloc[i]
            v=float(r["vwap"])
            c=float(r["close"])
            exit_sig=(side=="LONG" and c>=v) or (side=="SHORT" and c<=v)
            if not exit_sig:
                continue

            raw=float(g.iloc[i+1]["open"])
            exit_fill=_slip(raw,side,False)
            shares=max(1,int(SPEC["notional_per_trade"]/max(entry_fill,0.01)))
            gross=(exit_fill-entry_fill)*(1 if side=="LONG" else -1)*shares
            comm=2*SPEC["commission_per_share"]*shares
            net=gross-comm
            ret=net/(entry_fill*shares)
            trades.append({
                "symbol":sym,"date":str(d),"side":side,
                "entry_time":str(g.iloc[entry_i]["timestamp"]),
                "exit_time":str(g.iloc[i+1]["timestamp"]),
                "entry_price":round(entry_fill,4),"exit_price":round(exit_fill,4),
                "shares":shares,"exit_reason":"VWAP_REVERSION",
                "entry_signal_vwap":round(entry_vwap,4),
                "pnl_net":round(net,4),"return_pct":round(ret*100,6)
            })
            entered=False
            break

        if entered:
            r=g.iloc[-1]
            exit_fill=_slip(float(r["close"]),side,False)
            shares=max(1,int(SPEC["notional_per_trade"]/max(entry_fill,0.01)))
            gross=(exit_fill-entry_fill)*(1 if side=="LONG" else -1)*shares
            comm=2*SPEC["commission_per_share"]*shares
            net=gross-comm
            ret=net/(entry_fill*shares)
            trades.append({
                "symbol":sym,"date":str(d),"side":side,
                "entry_time":str(g.iloc[entry_i]["timestamp"]),
                "exit_time":str(r["timestamp"]),
                "entry_price":round(entry_fill,4),"exit_price":round(exit_fill,4),
                "shares":shares,"exit_reason":"EOD",
                "entry_signal_vwap":round(entry_vwap,4),
                "pnl_net":round(net,4),"return_pct":round(ret*100,6)
            })

        if progress_cb and (n%500==0 or n==total):
            progress_cb(p0+int((p1-p0)*n/total),f"{label}: {n}/{total} symbol-days")

    return pd.DataFrame(trades)

def _metrics(t):
    if t.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,
                "avg_trade_pct":0.0,"median_trade_pct":0.0,
                "max_dd_pct":0.0,"net_pnl":0.0,"long_trades":0,"short_trades":0}
    pnl=pd.to_numeric(t["pnl_net"],errors="coerce").fillna(0)
    rr=pd.to_numeric(t["return_pct"],errors="coerce").dropna()
    gp=float(pnl[pnl>0].sum()); gl=float(-pnl[pnl<0].sum())
    pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    eq=(1+rr/100.0).cumprod()
    dd=eq/eq.cummax()-1
    return {
        "trades":int(len(t)),
        "win_rate":round(float((pnl>0).mean()),4),
        "profit_factor":round(float(pf),4),
        "avg_trade_pct":round(float(rr.mean()),4),
        "median_trade_pct":round(float(rr.median()),4),
        "max_dd_pct":round(float(dd.min()*100),4),
        "net_pnl":round(float(pnl.sum()),2),
        "long_trades":int((t["side"]=="LONG").sum()),
        "short_trades":int((t["side"]=="SHORT").sum())
    }

def _bootstrap(t,n=2000,seed=20260901):
    if t.empty:
        return {"prob_pf_gt1":0.0,"prob_mean_return_gt0":0.0}
    pnl=pd.to_numeric(t["pnl_net"],errors="coerce").fillna(0).to_numpy(float)
    ret=pd.to_numeric(t["return_pct"],errors="coerce").fillna(0).to_numpy(float)
    rng=np.random.default_rng(seed); pfs=[]; means=[]
    for _ in range(n):
        idx=rng.integers(0,len(t),len(t))
        p=pnl[idx]; r=ret[idx]
        gp=p[p>0].sum(); gl=-p[p<0].sum()
        pfs.append(gp/gl if gl>0 else (20 if gp>0 else 0))
        means.append(r.mean())
    return {
        "prob_pf_gt1":round(float((np.array(pfs)>1).mean()),4),
        "prob_mean_return_gt0":round(float((np.array(means)>0).mean()),4),
        "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
        "mean_return_ci95_pct":[round(float(np.quantile(means,.025)),4),round(float(np.quantile(means,.975)),4)]
    }

def run_external_challenger3(progress_cb=None):
    manifest={"discovery":_manifest(DISCOVERY,"Discovery"),"true_oos":_manifest(TRUE_OOS,"True OOS")}
    for k,v in manifest.items():
        if not v.get("exists") or not v.get("required_columns_ok"):
            raise ValueError(f"{k} manifest/schema preflight failed: {v}")

    OUTDIR.mkdir(parents=True,exist_ok=True)
    if progress_cb: progress_cb(2,"Manifest PASS · source-documented VWAP mean reversion defaults")

    dx=_prepare(DISCOVERY,"Discovery")
    dt=_run_period(dx,"Discovery",progress_cb,5,45)
    ox=_prepare(TRUE_OOS,"True OOS")
    ot=_run_period(ox,"True OOS",progress_cb,50,90)

    dt.to_csv(OUTDIR/"discovery_trades.csv",index=False)
    ot.to_csv(OUTDIR/"true_oos_trades.csv",index=False)

    result={
        "study":"External Challenger #3 · VWAP Mean Reversion v1",
        "source_locked_spec":SPEC,
        "data_manifest":manifest,
        "source_fidelity_note":"Rules and defaults follow 555aaditya/Stock-Backtesting-Pipeline STRATEGIES.md: VWAP ±1.5 session standard deviations, volume >1.2x average, fade toward VWAP. The source docs do not specify a hard stop, so none is invented. We add common slippage/commission and next-bar execution to avoid look-ahead.",
        "discovery":{"metrics":_metrics(dt),"rows":int(len(dx))},
        "true_oos":{"metrics":_metrics(ot),"rows":int(len(ox)),"bootstrap":_bootstrap(ot)},
        "decision_rule":"No rescue tuning. PROMOTE only if True OOS has >=100 trades, PF>1, average trade return>0, bootstrap P(PF>1)>=80%, and both long/short have non-trivial counts. Otherwise DROP on our 125-stock pool."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        m=result["true_oos"]["metrics"]
        progress_cb(100,f"Challenger #3 complete · OOS {m['trades']} trades · PF {m['profit_factor']} · AvgTrade {m['avg_trade_pct']}%")
    return result
