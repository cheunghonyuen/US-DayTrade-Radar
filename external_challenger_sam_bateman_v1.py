
import json, math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/daytrade")
DISCOVERY = DATA_ROOT / "sip_data.csv"
OOS_CANDIDATES = [
    # Exact persistent True OOS path already used by app.py / V0.8 / setup tournament / QC benchmark.
    DATA_ROOT / "oos_v08" / "sip_oos_2024-08-30_2025-08-29.csv",
]
OUTDIR = DATA_ROOT / "external_challenger_sam_bateman_v1"

SPEC = {
    "source_repo": "sam-bateman/trading-orb",
    "source_config": "run_max_backtest.py validated params",
    "or_minutes": 20,
    "target_mult": 0.75,
    "stop_mult": 0.50,
    "vol_thresh": 1.20,
    "entry_start": 10.0,
    "entry_end": 11.5,
    "direction": "both",
    "risk_per_trade": 400.0,
    "max_positions": 3,
    "max_notional": 50000.0,
    "max_pct_account": 0.05,
    "daily_loss_limit": 600.0,
    "slippage_per_share": 0.01,
    "commission_per_share": 0.005,
    "flatten_time": 15.833,
    "relative_volume_lookback": 20,
    "breakeven_after_or_profit": True,
    "universe_constraint": "Current 125-stock persistent replay, not author's screened 20-stock universe",
    "parameter_tuning": "NONE"
}

REQ = {"symbol","timestamp","open","high","low","close","volume"}

def _locate_oos():
    p = OOS_CANDIDATES[0]
    if not p.exists():
        raise FileNotFoundError(
            f"Exact True OOS persistent replay missing: {p}. "
            "Do not redownload or guess another path; run dashboard Preflight to verify the mounted volume."
        )
    return p

def _manifest_probe(path, label):
    p = Path(path)
    info = {
        "label": label,
        "path": str(p),
        "exists": p.exists(),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 2) if p.exists() else 0.0,
    }
    if not p.exists():
        return info
    probe = pd.read_csv(p, nrows=5000)
    info["columns"] = list(probe.columns)
    info["required_columns_ok"] = bool(REQ.issubset(set(probe.columns)))
    if "symbol" in probe.columns:
        info["sample_symbols"] = int(probe["symbol"].astype(str).nunique())
    if "timestamp" in probe.columns:
        ts = pd.to_datetime(probe["timestamp"], errors="coerce", utc=True)
        info["sample_timestamp_min"] = str(ts.min())
        info["sample_timestamp_max"] = str(ts.max())
        info["invalid_timestamp_sample"] = int(ts.isna().sum())
    return info

def _prepare(path, label):
    x = pd.read_csv(path)
    miss = sorted(REQ - set(x.columns))
    if miss:
        raise ValueError(f"{label} missing columns {miss}")
    if x.duplicated(["symbol","timestamp"]).any():
        raise ValueError(f"{label} duplicate symbol/timestamp")
    ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"{label} invalid timestamps {int(ts.isna().sum())}")
    x["timestamp"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["symbol","timestamp","open","high","low","close","volume"]).copy()
    x["date"] = x["timestamp"].dt.date
    x["time_decimal"] = x["timestamp"].dt.hour + x["timestamp"].dt.minute/60.0
    x["time_key"] = x["timestamp"].dt.strftime("%H:%M")
    x = x[(x["time_decimal"] >= 9.5) & (x["time_decimal"] <= 16.0)].copy()
    x.sort_values(["symbol","timestamp"], inplace=True)

    # Author's relative volume: each bar vs rolling 20 observations at same time bucket,
    # min_periods=5, current bar included by the published source.
    x["rel_volume"] = (
        x.groupby(["symbol","time_key"], sort=False)["volume"]
         .transform(lambda s: s / (s.rolling(SPEC["relative_volume_lookback"], min_periods=5).mean() + 1.0))
    )

    # OR20 = first 20 minutes, 09:30 <= t < 09:50.
    opening = x[(x["time_decimal"] >= 9.5) & (x["time_decimal"] < 9.5 + SPEC["or_minutes"]/60.0)]
    ors = opening.groupby(["symbol","date"], sort=False).agg(
        or_high=("high","max"), or_low=("low","min")
    ).reset_index()
    ors["or_range"] = ors["or_high"] - ors["or_low"]
    x = x.merge(ors, on=["symbol","date"], how="left")

    return x

def _generate_signals(x):
    rows=[]
    for (sym,d), g in x.groupby(["symbol","date"], sort=False):
        g = g.sort_values("timestamp")
        if len(g) < 10:
            continue
        or_high = g["or_high"].iloc[0]
        or_low = g["or_low"].iloc[0]
        or_range = g["or_range"].iloc[0]
        if pd.isna(or_high) or pd.isna(or_low) or not np.isfinite(or_range) or or_range <= 0:
            continue
        long_trig=False; short_trig=False
        for r in g.itertuples(index=False):
            t=float(r.time_decimal)
            if t < SPEC["entry_start"] or t > SPEC["entry_end"]:
                continue
            rv=float(r.rel_volume) if pd.notna(r.rel_volume) else 1.0
            # Source final config has direction both, no extra vol-regime / prev-day / OR-quality filters.
            if (not long_trig) and float(r.close) > float(or_high) and rv >= SPEC["vol_thresh"]:
                ep=float(r.close)
                rows.append({
                    "symbol":sym,"date":d,"signal_time":r.timestamp,"direction":1,
                    "signal_price":ep,"target_price":ep + or_range*SPEC["target_mult"],
                    "stop_price":ep - or_range*SPEC["stop_mult"],"or_range":or_range,
                    "rel_volume":rv
                })
                long_trig=True
            if (not short_trig) and float(r.close) < float(or_low) and rv >= SPEC["vol_thresh"]:
                ep=float(r.close)
                rows.append({
                    "symbol":sym,"date":d,"signal_time":r.timestamp,"direction":-1,
                    "signal_price":ep,"target_price":ep - or_range*SPEC["target_mult"],
                    "stop_price":ep + or_range*SPEC["stop_mult"],"or_range":or_range,
                    "rel_volume":rv
                })
                short_trig=True
    return pd.DataFrame(rows)

@dataclass
class Position:
    symbol: str
    direction: int
    entry_time: object
    entry_price: float
    signal_price: float
    shares: int
    target_price: float
    stop_price: float
    or_range: float
    breakeven: bool = False

def _apply_slip(price, direction, is_entry):
    slip=SPEC["slippage_per_share"]
    return float(price) + (slip*direction if is_entry else -slip*direction)

def _run_engine(x, signals, progress_cb=None, p0=0, p1=100, label="period"):
    sig_by_key={}
    for rr in signals.to_dict("records"):
        sig_by_key.setdefault((rr["symbol"], pd.Timestamp(rr["signal_time"])), []).append(rr)

    # Bars globally chronological; stable symbol ordering.
    bars=x.sort_values(["timestamp","symbol"]).reset_index(drop=True)

    capital=100000.0
    open_pos=[]
    pending=[]  # entries fill only on next bar of same symbol
    traded_by_day={}
    daily_pnl={}
    halted={}
    closed=[]

    # Track previous processed bar timestamp per symbol so next same-symbol bar can fill.
    last_seen={}
    total=len(bars)

    def close_pos(pos, raw_exit, ts, reason):
        nonlocal capital
        fill=_apply_slip(raw_exit,pos.direction,False)
        gross=(fill-pos.entry_price)*pos.direction*pos.shares
        slip_cost=SPEC["slippage_per_share"]*2*pos.shares
        comm=SPEC["commission_per_share"]*2*pos.shares
        net=gross-slip_cost-comm
        capital += net
        d=pd.Timestamp(ts).date()
        daily_pnl[d]=daily_pnl.get(d,0.0)+net
        risk_per_share=abs(pos.signal_price-pos.stop_price)
        risk_dollars=risk_per_share*pos.shares if risk_per_share>0 else np.nan
        r_mult=net/risk_dollars if risk_dollars and np.isfinite(risk_dollars) and risk_dollars>0 else np.nan
        closed.append({
            "symbol":pos.symbol,"direction":"LONG" if pos.direction==1 else "SHORT",
            "entry_time":pos.entry_time,"exit_time":ts,"entry_price":pos.entry_price,
            "exit_price":fill,"shares":pos.shares,"exit_reason":reason,
            "pnl_net":net,"pnl_r":r_mult
        })

    current_date=None
    for i,b in bars.iterrows():
        ts=b["timestamp"]; d=b["date"]; sym=b["symbol"]

        if current_date is None or d != current_date:
            # Any stale positions should not exist because of flatten; defensively close at last known close.
            current_date=d

        # 1) exits for open positions of this symbol
        survivors=[]
        for pos in open_pos:
            if pos.symbol != sym:
                survivors.append(pos); continue

            t=float(b["time_decimal"])
            done=False
            if t >= SPEC["flatten_time"]:
                close_pos(pos,float(b["close"]),ts,"time_exit"); done=True
            elif pos.direction==1 and float(b["low"]) <= pos.stop_price:
                close_pos(pos,pos.stop_price,ts,"stop_loss"); done=True
            elif pos.direction==-1 and float(b["high"]) >= pos.stop_price:
                close_pos(pos,pos.stop_price,ts,"stop_loss"); done=True
            elif pos.direction==1 and float(b["high"]) >= pos.target_price:
                close_pos(pos,pos.target_price,ts,"target_hit"); done=True
            elif pos.direction==-1 and float(b["low"]) <= pos.target_price:
                close_pos(pos,pos.target_price,ts,"target_hit"); done=True

            if not done:
                if SPEC["breakeven_after_or_profit"] and (not pos.breakeven):
                    unreal=(float(b["close"])-pos.entry_price)*pos.direction
                    if unreal >= pos.or_range:
                        pos.stop_price=pos.entry_price + 0.01*pos.direction
                        pos.breakeven=True
                survivors.append(pos)
        open_pos=survivors

        # 2) pending entries for this symbol fill at this bar open, but only if this is later than signal bar.
        still_pending=[]
        for ent in pending:
            if ent["symbol"] != sym:
                still_pending.append(ent); continue
            if pd.Timestamp(ts) <= pd.Timestamp(ent["signal_time"]):
                still_pending.append(ent); continue

            # processed once on next bar then removed whether accepted or rejected
            if halted.get(d,False):
                continue
            if daily_pnl.get(d,0.0) <= -SPEC["daily_loss_limit"]:
                halted[d]=True
                continue
            if len(open_pos) >= SPEC["max_positions"]:
                continue
            traded=traded_by_day.setdefault(d,set())
            if sym in traded:
                continue

            raw_entry=float(b["open"])
            fill=_apply_slip(raw_entry,ent["direction"],True)
            risk_ps=abs(fill-ent["stop_price"])
            if risk_ps<=0:
                continue
            shares=int(SPEC["risk_per_trade"]/risk_ps)
            shares=min(shares,int(SPEC["max_notional"]/fill))
            shares=min(shares,int((capital*SPEC["max_pct_account"])/fill))
            if shares < 1:
                continue

            open_pos.append(Position(
                symbol=sym,direction=ent["direction"],entry_time=ts,entry_price=fill,
                signal_price=ent["signal_price"],shares=shares,target_price=ent["target_price"],
                stop_price=ent["stop_price"],or_range=ent["or_range"]
            ))
            traded.add(sym)
        pending=still_pending

        # 3) queue signals occurring on this bar for next bar open.
        for ent in sig_by_key.get((sym,pd.Timestamp(ts)),[]):
            pending.append(ent)

        if progress_cb and (i%100000==0 or i==total-1):
            pct=p0+int((p1-p0)*(i+1)/max(total,1))
            progress_cb(min(pct,p1),f"{label}: event replay {i+1:,}/{total:,}")

    # Defensive close any leftovers at their symbol's final bar close.
    if open_pos:
        lastbar=x.sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")
        for pos in list(open_pos):
            if pos.symbol in lastbar.index:
                rb=lastbar.loc[pos.symbol]
                close_pos(pos,float(rb["close"]),rb["timestamp"],"final_close")

    return pd.DataFrame(closed)

def _metrics(t):
    if t.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,
                "median_r":0.0,"max_dd_r":0.0,"net_pnl":0.0}
    pnl=pd.to_numeric(t["pnl_net"],errors="coerce").fillna(0.0)
    r=pd.to_numeric(t["pnl_r"],errors="coerce").dropna()
    gp=float(pnl[pnl>0].sum()); gl=float(-pnl[pnl<0].sum())
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    eq=pnl.cumsum(); dd=eq-eq.cummax()
    req=r.cumsum() if len(r) else pd.Series(dtype=float)
    rdd=req-req.cummax() if len(req) else pd.Series(dtype=float)
    return {
        "trades":int(len(t)),
        "win_rate":round(float((pnl>0).mean()),4),
        "profit_factor":round(float(pf),4),
        "avg_r":round(float(r.mean()),4) if len(r) else 0.0,
        "median_r":round(float(r.median()),4) if len(r) else 0.0,
        "max_dd_r":round(float(rdd.min()),4) if len(rdd) else 0.0,
        "net_pnl":round(float(pnl.sum()),2),
        "long_trades":int((t["direction"]=="LONG").sum()),
        "short_trades":int((t["direction"]=="SHORT").sum()),
    }

def _bootstrap(t,n=2000,seed=20260901):
    if t.empty:
        return {"prob_pf_gt1":0.0,"prob_avg_r_gt0":0.0}
    r=pd.to_numeric(t["pnl_r"],errors="coerce").dropna().to_numpy(float)
    if len(r)==0: return {"prob_pf_gt1":0.0,"prob_avg_r_gt0":0.0}
    rng=np.random.default_rng(seed)
    pfs=[]; avgs=[]
    for _ in range(n):
        s=r[rng.integers(0,len(r),len(r))]
        gp=s[s>0].sum(); gl=-s[s<0].sum()
        pfs.append(gp/gl if gl>0 else (20.0 if gp>0 else 0.0))
        avgs.append(s.mean())
    return {
        "prob_pf_gt1":round(float((np.array(pfs)>1).mean()),4),
        "prob_avg_r_gt0":round(float((np.array(avgs)>0).mean()),4),
        "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
        "avg_r_ci95":[round(float(np.quantile(avgs,.025)),4),round(float(np.quantile(avgs,.975)),4)]
    }

def _run_period(path,label,progress_cb,p0,p1):
    if progress_cb: progress_cb(p0,f"{label}: schema/timezone/feature preflight")
    x=_prepare(path,label)
    if progress_cb: progress_cb(p0+5,f"{label}: {len(x):,} regular rows · {x['symbol'].nunique()} symbols")
    sig=_generate_signals(x)
    if progress_cb: progress_cb(p0+10,f"{label}: source-rule signals {len(sig):,}")
    t=_run_engine(x,sig,progress_cb,p0+10,p1,label)
    return x,sig,t

def run_external_challenger(progress_cb=None):
    if not DISCOVERY.exists():
        raise FileNotFoundError(f"Discovery persistent replay missing: {DISCOVERY}")
    oos=_locate_oos()
    OUTDIR.mkdir(parents=True,exist_ok=True)

    manifest = {
        "discovery": _manifest_probe(DISCOVERY, "Discovery"),
        "true_oos": _manifest_probe(oos, "True OOS"),
    }
    for k, v in manifest.items():
        if not v.get("exists") or not v.get("required_columns_ok"):
            raise ValueError(f"{k} manifest/schema preflight failed: {v}")

    if progress_cb:
        progress_cb(
            1,
            f"Data Manifest PASS · D {manifest['discovery']['size_mb']} MB · "
            f"OOS {manifest['true_oos']['size_mb']} MB · source-locked, no tuning"
        )
    dx,ds,dt=_run_period(DISCOVERY,"Discovery",progress_cb,2,45)
    ox,osig,ot=_run_period(oos,"True OOS",progress_cb,47,90)

    dt.to_csv(OUTDIR/"discovery_trades.csv",index=False)
    ot.to_csv(OUTDIR/"true_oos_trades.csv",index=False)

    result={
        "study":"External Challenger #1 · sam-bateman/trading-orb v1",
        "data_manifest": manifest,
        "source_locked_spec":SPEC,
        "fidelity":{
            "matched":["OR20","breakout-bar close signal","RVOL >=1.2 at same time bucket rolling20",
                       "10:00-11:30 entry window","both directions","target .75x OR","stop .50x OR",
                       "next-bar-open fill","$0.01/share slippage","$0.005/share commission",
                       "$400 risk/trade","max 3 positions","$600 daily loss limit","15:50 flatten",
                       "breakeven stop after close reaches +1 OR"],
            "constraint":"Uses our current 125-stock replay rather than author's screened 20-stock universe."
        },
        "discovery":{"metrics":_metrics(dt),"signals":int(len(ds)),
                     "rows":int(len(dx))},
        "true_oos":{"metrics":_metrics(ot),"signals":int(len(osig)),
                    "rows":int(len(ox)),"bootstrap":_bootstrap(ot)},
        "author_claim_reference":{
            "trades":4292,"win_rate_pct":50.3,"profit_factor":1.31,
            "sharpe":2.47,"max_drawdown_pct":1.27
        },
        "decision_rule":"No rescue tuning. Promote to a separate frozen-forward challenger only if True OOS PF>1, AvgR>0, bootstrap P(PF>1)>=80%, and both long/short have non-trivial trade counts. Otherwise DROP on our 125-stock pool."
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        m=result["true_oos"]["metrics"]
        progress_cb(100,f"Challenger complete · OOS PF {m['profit_factor']} · AvgR {m['avg_r']}")
    return result
