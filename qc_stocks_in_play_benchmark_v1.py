
import json
from pathlib import Path
import numpy as np
import pandas as pd

SPEC = {
    "source": "QuantConnect research: Opening Range Breakout for Stocks in Play",
    "opening_range_minutes": 5,
    "indicator_period_days": 14,
    "min_price": 5.0,
    "min_atr_dollars": 0.50,
    "min_relative_volume": 1.0,
    "max_positions": 20,
    "stop_atr_fraction": 0.10,
    "stop_loss_risk_size": 0.01,
    "position_weight_cap": 0.05,
    "exit": "end_of_day",
    "universe_constraint": "Current persistent 125-stock pool only; original article uses a much larger liquid-US-equity universe."
}

REQ = {"symbol","timestamp","open","high","low","close","volume"}

def _pf(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    return gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

def _summary(t):
    if t.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,
                "median_r":0.0,"max_dd_r":0.0,"stop_rate":0.0}
    r = pd.to_numeric(t["pnl_r"], errors="coerce").fillna(0.0)
    eq = r.cumsum()
    dd = eq - eq.cummax()
    return {
        "trades": int(len(t)),
        "win_rate": round(float((r > 0).mean()),4),
        "profit_factor": round(float(_pf(r)),4),
        "avg_r": round(float(r.mean()),4),
        "median_r": round(float(r.median()),4),
        "max_dd_r": round(float(dd.min()),4),
        "stop_rate": round(float(t["exit_reason"].astype(str).eq("STOP").mean()),4),
    }

def _bootstrap(t, n=2000, seed=20260901):
    if t.empty:
        return {"prob_pf_gt1":0.0,"prob_avg_r_gt0":0.0,"pf_ci95":[0,0],"avg_r_ci95":[0,0]}
    r = pd.to_numeric(t["pnl_r"],errors="coerce").fillna(0.0).to_numpy(float)
    rng = np.random.default_rng(seed)
    pfs = np.empty(n); av = np.empty(n); N=len(r)
    for i in range(n):
        s = r[rng.integers(0,N,N)]
        av[i] = s.mean()
        gp=s[s>0].sum(); gl=-s[s<0].sum()
        pfs[i] = gp/gl if gl>0 else (20.0 if gp>0 else 0.0)
    return {
        "prob_pf_gt1":round(float((pfs>1).mean()),4),
        "prob_avg_r_gt0":round(float((av>0).mean()),4),
        "pf_ci95":[round(float(np.quantile(pfs,.025)),4),round(float(np.quantile(pfs,.975)),4)],
        "avg_r_ci95":[round(float(np.quantile(av,.025)),4),round(float(np.quantile(av,.975)),4)]
    }

def _prep(path, progress_cb=None, p0=0, p1=35, label="period"):
    x = pd.read_csv(path)
    missing = sorted(REQ-set(x.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    if x.duplicated(["symbol","timestamp"]).any():
        raise ValueError(f"{label} duplicate symbol/timestamp rows found")

    ts = pd.to_datetime(x["timestamp"], errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"{label} invalid timestamps: {int(ts.isna().sum())}")
    x["timestamp"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    x["time"] = x["timestamp"].dt.strftime("%H:%M")
    x = x[(x["time"] >= "09:30") & (x["time"] <= "15:55")].copy()
    x.sort_values(["symbol","timestamp"], inplace=True)

    if progress_cb:
        progress_cb(p0+4, f"{label}: session/schema OK · rows {len(x):,}")

    # Daily bars and prior-day ATR14.
    daily = x.groupby(["symbol","date"],sort=True).agg(
        dopen=("open","first"), dhigh=("high","max"), dlow=("low","min"),
        dclose=("close","last")
    ).reset_index()
    daily["prev_close"] = daily.groupby("symbol")["dclose"].shift(1)
    daily["tr"] = np.maximum(
        daily["dhigh"]-daily["dlow"],
        np.maximum((daily["dhigh"]-daily["prev_close"]).abs(),
                   (daily["dlow"]-daily["prev_close"]).abs())
    )
    # Use only fully completed previous sessions at 09:35.
    daily["atr14"] = daily.groupby("symbol")["tr"].transform(
        lambda s: s.shift(1).rolling(SPEC["indicator_period_days"], min_periods=SPEC["indicator_period_days"]).mean()
    )

    # First 5-min opening bar and prior-14 opening-volume SMA.
    first = x.groupby(["symbol","date"],sort=True).first().reset_index()
    first = first[["symbol","date","timestamp","open","high","low","close","volume"]].copy()
    first["openvol_sma14"] = first.groupby("symbol")["volume"].transform(
        lambda s: s.shift(1).rolling(SPEC["indicator_period_days"], min_periods=SPEC["indicator_period_days"]).mean()
    )
    first["rvol"] = first["volume"] / first["openvol_sma14"]
    first = first.merge(daily[["symbol","date","atr14"]], on=["symbol","date"], how="left")

    # Article filters on our constrained 125-stock universe.
    cand = first[
        (first["close"] > SPEC["min_price"]) &
        (first["atr14"] > SPEC["min_atr_dollars"]) &
        (first["rvol"] > SPEC["min_relative_volume"])
    ].copy()
    cand["rank"] = cand.groupby("date")["rvol"].rank(method="first",ascending=False)
    cand = cand[cand["rank"] <= SPEC["max_positions"]].copy()
    cand.sort_values(["date","rank"],inplace=True)

    if progress_cb:
        progress_cb(p0+12, f"{label}: candidates {len(cand):,} after Price/ATR/RVOL/top20")

    # Index sessions for fast lookup.
    sessions = {(s,d):g.reset_index(drop=True) for (s,d),g in x.groupby(["symbol","date"],sort=False)}

    strict_rows=[]
    delayed_rows=[]
    total=len(cand)

    for i,row in enumerate(cand.itertuples(index=False),1):
        g=sessions.get((row.symbol,row.date))
        if g is None or len(g)<2:
            continue

        # Order is placed at 09:35 after opening 5-min bar is known.
        post = g[g["timestamp"] > row.timestamp].copy()
        if post.empty:
            continue

        if float(row.close) > float(row.open):
            side=1
            entry=float(row.high)
            stop=entry-SPEC["stop_atr_fraction"]*float(row.atr14)
            hit=post[post["high"]>=entry]
        elif float(row.close) < float(row.open):
            side=-1
            entry=float(row.low)
            stop=entry+SPEC["stop_atr_fraction"]*float(row.atr14)
            hit=post[post["low"]<=entry]
        else:
            continue
        if hit.empty:
            continue

        ent_idx=hit.index[0]
        ent_bar=post.loc[ent_idx]
        risk=abs(entry-stop)
        if risk<=0:
            continue

        # STRICT: same 5-min entry bar may also hit stop => stop-first conservative bound.
        same_stop = (side==1 and float(ent_bar["low"])<=stop) or (side==-1 and float(ent_bar["high"])>=stop)
        if same_stop:
            strict_exit=stop; strict_reason="STOP"
        else:
            after=post.loc[post.index>=ent_idx]
            strict_exit=None; strict_reason=None
            for b in after.itertuples(index=False):
                stopped=(side==1 and float(b.low)<=stop) or (side==-1 and float(b.high)>=stop)
                if stopped:
                    strict_exit=stop; strict_reason="STOP"; break
            if strict_exit is None:
                strict_exit=float(g.iloc[-1]["close"]); strict_reason="EOD"

        # DELAYED-STOP bound: stop becomes active from next 5-min bar after entry.
        later=post.loc[post.index>ent_idx]
        delayed_exit=None; delayed_reason=None
        for b in later.itertuples(index=False):
            stopped=(side==1 and float(b.low)<=stop) or (side==-1 and float(b.high)>=stop)
            if stopped:
                delayed_exit=stop; delayed_reason="STOP"; break
        if delayed_exit is None:
            delayed_exit=float(g.iloc[-1]["close"]); delayed_reason="EOD"

        base={
            "symbol":row.symbol,"date":row.date,"rvol":round(float(row.rvol),4),
            "atr14":round(float(row.atr14),4),"side":"LONG" if side==1 else "SHORT",
            "entry_time":str(ent_bar["timestamp"]),"entry":round(entry,4),"stop":round(stop,4)
        }
        sr=(strict_exit-entry)*side/risk
        dr=(delayed_exit-entry)*side/risk
        strict_rows.append({**base,"exit":round(strict_exit,4),"exit_reason":strict_reason,"pnl_r":round(float(sr),4)})
        delayed_rows.append({**base,"exit":round(delayed_exit,4),"exit_reason":delayed_reason,"pnl_r":round(float(dr),4)})

        if progress_cb and (i%250==0 or i==total):
            pct=p0+12+int((p1-p0-12)*i/max(total,1))
            progress_cb(min(pct,p1),f"{label}: replay {i}/{total}")

    return pd.DataFrame(strict_rows), pd.DataFrame(delayed_rows), {
        "regular_rows":int(len(x)),
        "sessions":int(x[["symbol","date"]].drop_duplicates().shape[0]),
        "eligible_candidates":int(len(cand)),
        "symbols":int(x["symbol"].nunique())
    }

def run_qc_stocks_in_play_benchmark(discovery_path,oos_path,outdir,progress_cb=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)

    if progress_cb: progress_cb(2,"QuantConnect article-rule benchmark · source-fidelity preflight")
    ds,dd,dmeta=_prep(discovery_path,progress_cb,2,43,"Discovery")
    if progress_cb: progress_cb(47,"True OOS")
    os,od,ometa=_prep(oos_path,progress_cb,47,88,"True OOS")

    # Persist all outputs.
    ds.to_csv(out/"qc_discovery_strict.csv",index=False)
    dd.to_csv(out/"qc_discovery_delayed_stop.csv",index=False)
    os.to_csv(out/"qc_oos_strict.csv",index=False)
    od.to_csv(out/"qc_oos_delayed_stop.csv",index=False)

    if progress_cb: progress_cb(91,"Metrics + bootstrap")
    result={
        "study":"QuantConnect Stocks-in-Play ORB External Benchmark v1",
        "spec":SPEC,
        "fidelity":{
            "faithful_rules":["5-min opening bar","RVOL=opening volume / prior-14 opening-volume SMA",
                             "top20 RVOL","Price>$5","ATR14>$0.50","long if green opening bar / short if red",
                             "stop-market at opening high/low","stop distance=10% ATR","EOD liquidation"],
            "known_constraint":"Universe is current 125 stocks, not QuantConnect's 1000-liquid-stock monthly universe.",
            "microstructure_bound":"Because source data is 5-min, both same-entry-bar stop-first (strict) and next-bar stop activation (delayed) are reported."
        },
        "discovery":{
            "strict":_summary(ds),"delayed_stop":_summary(dd),"meta":dmeta,
            "bootstrap_strict":_bootstrap(ds)
        },
        "true_oos":{
            "strict":_summary(os),"delayed_stop":_summary(od),"meta":ometa,
            "bootstrap_strict":_bootstrap(os)
        },
        "decision_rule":"High-value benchmark only. Do not tune. If both OOS bounds are weak (PF<1 and AvgR<0), drop this external strategy on the 125-stock pool. If both are positive, promote to engine-parity stage. If bounds disagree, microstructure resolution is material and must be tested in LEAN/Backtrader before any strategy conclusion."
    }
    (out/"qc_stocks_in_play_benchmark_v1.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:progress_cb(100,"QuantConnect external benchmark complete")
    return result
