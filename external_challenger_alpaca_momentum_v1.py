
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/daytrade")
DISCOVERY = DATA_ROOT / "sip_data.csv"
TRUE_OOS = DATA_ROOT / "oos_v08" / "sip_oos_2024-08-30_2025-08-29.csv"
OUTDIR = DATA_ROOT / "external_challenger_alpaca_momentum_v1"

REQ = {"symbol","timestamp","open","high","low","close","volume"}

# Source-locked signal logic from alpacahq/Momentum-Trading-Example.
# Common execution frictions are added as a conservative comparison overlay.
SPEC = {
    "source_repo": "alpacahq/Momentum-Trading-Example",
    "family": "intraday momentum",
    "long_only": True,
    "min_share_price": 2.0,
    "max_share_price": 13.0,
    "min_prev_day_dollar_volume": 500000.0,
    "universe_change_filter_pct": 3.5,
    "entry_change_filter_pct": 4.0,
    "first_15m_breakout": True,
    "entry_window_minutes_after_open": [15, 60],
    "min_intraday_volume": 30000.0,
    "macd_fast_slow_1": [12, 26],
    "macd_1_positive": True,
    "macd_1_three_bar_increasing": True,
    "macd_fast_slow_2": [40, 60],
    "macd_2_positive_and_rising": True,
    "fallback_stop_pct": 0.95,
    "target_r": 3.0,
    "exit_macd_fast_slow": [13, 21],
    "eod_liquidate_last_minutes": 15,
    "risk_fraction": 0.001,
    "comparison_slippage_per_share": 0.01,
    "comparison_commission_per_share": 0.005,
    "parameter_tuning": "NONE",
    "universe_constraint": "Same persistent 125-stock pool; original code dynamically scans tradable US stocks."
}

def _manifest(path, label):
    p=Path(path)
    info={"label":label,"path":str(p),"exists":p.exists(),
          "size_mb":round(p.stat().st_size/1024/1024,2) if p.exists() else 0.0}
    if not p.exists():
        return info
    q=pd.read_csv(p,nrows=5000)
    info["columns"]=list(q.columns)
    info["required_columns_ok"]=bool(REQ.issubset(set(q.columns)))
    if "timestamp" in q:
        ts=pd.to_datetime(q["timestamp"],errors="coerce",utc=True)
        info["sample_timestamp_min"]=str(ts.min())
        info["sample_timestamp_max"]=str(ts.max())
    return info

def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _macd_line(close, fast, slow):
    return _ema(close, fast)-_ema(close, slow)

def _prepare(path,label):
    x=pd.read_csv(path)
    miss=sorted(REQ-set(x.columns))
    if miss: raise ValueError(f"{label} missing {miss}")
    if x.duplicated(["symbol","timestamp"]).any():
        raise ValueError(f"{label} duplicate symbol/timestamp")
    ts=pd.to_datetime(x["timestamp"],errors="coerce",utc=True)
    if ts.isna().any():
        raise ValueError(f"{label} invalid timestamps {int(ts.isna().sum())}")
    x["timestamp"]=ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["symbol","timestamp","open","high","low","close","volume"]).copy()
    x["date"]=x["timestamp"].dt.date
    x["minute"]=x["timestamp"].dt.hour*60+x["timestamp"].dt.minute
    x=x[(x["minute"]>=570)&(x["minute"]<=955)].copy()
    x.sort_values(["symbol","timestamp"],inplace=True)

    # Previous-session close and volume for dynamic source universe filters.
    daily=x.groupby(["symbol","date"],sort=True).agg(
        dclose=("close","last"), dvol=("volume","sum")
    ).reset_index()
    daily["prev_close"]=daily.groupby("symbol")["dclose"].shift(1)
    daily["prev_vol"]=daily.groupby("symbol")["dvol"].shift(1)
    x=x.merge(daily[["symbol","date","prev_close","prev_vol"]],
              on=["symbol","date"],how="left")
    return x

def _recent_stop(g, upto_idx, current_close):
    # Source: session lows resampled to 5m, last local minimum - $0.01;
    # persistent replay is 5m, so each bar low is already a 5m low.
    lows=g.loc[:upto_idx,"low"].astype(float).to_numpy()
    if len(lows)>=3:
        d=np.diff(lows)
        idx=np.where((d[:-1]<=0)&(d[1:]>0))[0]+1
        if len(idx)>0:
            return float(lows[idx[-1]])-0.01
    return float(current_close)*SPEC["fallback_stop_pct"]

def _buy_limit_fill(nextbar, limit):
    # Buy limit: gap below limit gets open; otherwise fill at limit only if low touches.
    o=float(nextbar["open"]); lo=float(nextbar["low"])
    if o<=limit: return o
    if lo<=limit: return float(limit)
    return None

def _sell_limit_fill(nextbar, limit):
    # Sell limit: gap above limit gets open; otherwise fill at limit only if high touches.
    o=float(nextbar["open"]); hi=float(nextbar["high"])
    if o>=limit: return o
    if hi>=limit: return float(limit)
    return None

def _simulate_period(x,label,progress_cb=None,p0=0,p1=100):
    trades=[]
    sessions=list(x.groupby(["symbol","date"],sort=False))
    total=max(len(sessions),1)

    for n,((sym,d),g0) in enumerate(sessions,1):
        g=g0.sort_values("timestamp").reset_index(drop=True).copy()
        if len(g)<15: 
            continue
        pc=g["prev_close"].iloc[0]; pv=g["prev_vol"].iloc[0]
        if pd.isna(pc) or pd.isna(pv) or float(pc)<=0:
            continue

        # Source's initial dynamic ticker scan:
        # price $2-$13, prev-day dollar volume >$500k, today's change >=3.5%.
        first_price=float(g.iloc[0]["open"])
        if not (SPEC["min_share_price"]<=first_price<=SPEC["max_share_price"]):
            continue
        if float(pv)*first_price <= SPEC["min_prev_day_dollar_volume"]:
            continue

        # Indicators computed only from bars available up to current point.
        close=g["close"].astype(float)
        m1=_macd_line(close,12,26)
        m2=_macd_line(close,40,60)
        mex=_macd_line(close,13,21)
        cumvol=g["volume"].astype(float).cumsum()
        high15=float(g[g["minute"]<585]["high"].max())  # 09:30,35,40 bars

        entered=False
        entry_i=None
        entry_fill=None
        stop=None
        target=None
        signal_price=None

        for i in range(len(g)-1):
            r=g.iloc[i]
            minute=int(r["minute"])
            daily_change=(float(r["close"])-float(pc))/float(pc)

            if not entered:
                # Original buys only during 15-60 minutes after open.
                if not (585 < minute < 630):
                    continue
                if daily_change < 0.035:  # would not be in source ticker list
                    continue
                if not (
                    daily_change > 0.04 and
                    float(r["close"]) > high15 and
                    float(cumvol.iloc[i]) > SPEC["min_intraday_volume"]
                ):
                    continue
                if i<2 or pd.isna(m1.iloc[i]) or pd.isna(m2.iloc[i]):
                    continue
                if not (float(m1.iloc[i])>=0 and float(m1.iloc[i-2])<float(m1.iloc[i-1])<float(m1.iloc[i])):
                    continue
                if not (float(m2.iloc[i])>=0 and i>=1 and float(m2.iloc[i])-float(m2.iloc[i-1])>=0):
                    continue

                signal_price=float(r["close"])
                raw_stop=_recent_stop(g,i,signal_price)
                if raw_stop>=signal_price:
                    continue
                limit=signal_price
                fill=_buy_limit_fill(g.iloc[i+1],limit)
                if fill is None:
                    continue
                entry_fill=float(fill)+SPEC["comparison_slippage_per_share"]
                stop=float(raw_stop)
                target=signal_price+(signal_price-stop)*SPEC["target_r"]
                entry_i=i+1
                entered=True
                continue

            # Once entered, evaluate exit at close, submit a sell-limit at that close;
            # final 15 min is market liquidation.
            if minute>=945:
                raw_exit=float(r["close"])
                exit_fill=raw_exit-SPEC["comparison_slippage_per_share"]
                reason="EOD"
                exit_i=i
            else:
                mx=float(mex.iloc[i]) if pd.notna(mex.iloc[i]) else np.nan
                exit_signal = (
                    float(r["close"])<=stop or
                    (float(r["close"])>=target and np.isfinite(mx) and mx<=0) or
                    (float(r["close"])<=signal_price and np.isfinite(mx) and mx<=0)
                )
                if not exit_signal:
                    continue
                lim=float(r["close"])
                sf=_sell_limit_fill(g.iloc[i+1],lim)
                if sf is None:
                    continue
                exit_fill=float(sf)-SPEC["comparison_slippage_per_share"]
                reason=("STOP_OR_MOMENTUM_EXIT" if float(r["close"])<=stop
                        else "TARGET_MACD_EXIT" if float(r["close"])>=target
                        else "COST_BASIS_MACD_EXIT")
                exit_i=i+1

            risk_ps=max(entry_fill-stop,1e-9)
            # $100k account, source risk=.001 => $100 risk; quantity from source.
            shares=max(1,int((100000.0*SPEC["risk_fraction"])//risk_ps))
            gross=(exit_fill-entry_fill)*shares
            comm=2*SPEC["comparison_commission_per_share"]*shares
            net=gross-comm
            r_mult=net/(risk_ps*shares)
            trades.append({
                "symbol":sym,"date":str(d),"entry_time":str(g.iloc[entry_i]["timestamp"]),
                "exit_time":str(g.iloc[exit_i]["timestamp"]),"entry_price":round(entry_fill,4),
                "exit_price":round(exit_fill,4),"stop":round(stop,4),"target":round(target,4),
                "shares":shares,"exit_reason":reason,"pnl_net":round(net,4),"pnl_r":round(r_mult,4)
            })
            entered=False
            break  # source won't repurchase after an existing bought position within our session replay

        # defensive EOD if still open
        if entered:
            r=g.iloc[-1]
            exit_fill=float(r["close"])-SPEC["comparison_slippage_per_share"]
            risk_ps=max(entry_fill-stop,1e-9)
            shares=max(1,int((100000.0*SPEC["risk_fraction"])//risk_ps))
            net=(exit_fill-entry_fill)*shares-2*SPEC["comparison_commission_per_share"]*shares
            trades.append({
                "symbol":sym,"date":str(d),"entry_time":str(g.iloc[entry_i]["timestamp"]),
                "exit_time":str(r["timestamp"]),"entry_price":round(entry_fill,4),
                "exit_price":round(exit_fill,4),"stop":round(stop,4),"target":round(target,4),
                "shares":shares,"exit_reason":"EOD_FINAL","pnl_net":round(net,4),
                "pnl_r":round(net/(risk_ps*shares),4)
            })

        if progress_cb and (n%500==0 or n==total):
            progress_cb(p0+int((p1-p0)*n/total),f"{label}: {n}/{total} symbol-days")

    return pd.DataFrame(trades)

def _metrics(t):
    if t.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,
                "median_r":0.0,"max_dd_r":0.0,"net_pnl":0.0}
    pnl=pd.to_numeric(t["pnl_net"],errors="coerce").fillna(0)
    rr=pd.to_numeric(t["pnl_r"],errors="coerce").dropna()
    gp=float(pnl[pnl>0].sum()); gl=float(-pnl[pnl<0].sum())
    pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    req=rr.cumsum(); dd=req-req.cummax()
    return {"trades":int(len(t)),"win_rate":round(float((pnl>0).mean()),4),
            "profit_factor":round(float(pf),4),"avg_r":round(float(rr.mean()),4),
            "median_r":round(float(rr.median()),4),
            "max_dd_r":round(float(dd.min()),4),"net_pnl":round(float(pnl.sum()),2)}

def _bootstrap(t,n=2000,seed=20260901):
    if t.empty: return {"prob_pf_gt1":0.0,"prob_avg_r_gt0":0.0}
    r=pd.to_numeric(t["pnl_r"],errors="coerce").dropna().to_numpy(float)
    rng=np.random.default_rng(seed); pfs=[]; av=[]
    for _ in range(n):
        s=r[rng.integers(0,len(r),len(r))]
        gp=s[s>0].sum(); gl=-s[s<0].sum()
        pfs.append(gp/gl if gl>0 else (20 if gp>0 else 0))
        av.append(s.mean())
    return {"prob_pf_gt1":round(float((np.array(pfs)>1).mean()),4),
            "prob_avg_r_gt0":round(float((np.array(av)>0).mean()),4),
            "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
            "avg_r_ci95":[round(float(np.quantile(av,.025)),4),round(float(np.quantile(av,.975)),4)]}

def run_alpaca_momentum_challenger(progress_cb=None):
    manifest={"discovery":_manifest(DISCOVERY,"Discovery"),"true_oos":_manifest(TRUE_OOS,"True OOS")}
    for k,v in manifest.items():
        if not v.get("exists") or not v.get("required_columns_ok"):
            raise ValueError(f"{k} manifest/schema preflight failed: {v}")
    OUTDIR.mkdir(parents=True,exist_ok=True)
    if progress_cb: progress_cb(2,"Manifest PASS · source-locked Alpaca momentum rules")

    dx=_prepare(DISCOVERY,"Discovery")
    dt=_simulate_period(dx,"Discovery",progress_cb,5,45)
    if progress_cb: progress_cb(48,"True OOS preflight")
    ox=_prepare(TRUE_OOS,"True OOS")
    ot=_simulate_period(ox,"True OOS",progress_cb,50,90)

    dt.to_csv(OUTDIR/"discovery_trades.csv",index=False)
    ot.to_csv(OUTDIR/"true_oos_trades.csv",index=False)

    result={
        "study":"External Challenger #2 · Alpaca Momentum Trading Example v1",
        "source_locked_spec":SPEC,
        "data_manifest":manifest,
        "source_fidelity_note":"Signal/filter/stop/target/exit logic follows Alpaca example. Our common $0.01/share slippage + $0.005/share commission is added conservatively because the source example itself does not model these frictions.",
        "discovery":{"metrics":_metrics(dt),"rows":int(len(dx))},
        "true_oos":{"metrics":_metrics(ot),"rows":int(len(ox)),"bootstrap":_bootstrap(ot)},
        "decision_rule":"No rescue tuning. PROMOTE only if True OOS has >=100 trades, PF>1, AvgR>0, and bootstrap P(PF>1)>=80%. Otherwise DROP on our 125-stock pool."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        m=result["true_oos"]["metrics"]
        progress_cb(100,f"Alpaca Momentum complete · OOS trades {m['trades']} · PF {m['profit_factor']} · AvgR {m['avg_r']}")
    return result
