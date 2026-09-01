
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/daytrade")
DISCOVERY = DATA_ROOT / "sip_data.csv"
TRUE_OOS = DATA_ROOT / "oos_v08" / "sip_oos_2024-08-30_2025-08-29.csv"
OUTDIR = DATA_ROOT / "alpaca_macd_source_parity_v1"

REQ = {"symbol","timestamp","open","high","low","close","volume"}

SOURCE_FACTS = {
    "history_loader": "get_1000m_history_data(symbols): Polygon historic_agg(size='minute', limit=1000)",
    "macd_entry_1": "macd(close.dropna(), n_fast=12, n_slow=26); require >=0 and hist[-3] < hist[-2] < hist[-1]",
    "macd_entry_2": "macd(close.dropna(), n_fast=40, n_slow=60); require >=0 and np.diff(hist)[-1] >=0",
    "streaming": "1-second bars update current minute_history; incoming 1-minute bars replace aggregate",
    "important_parity_point": "MACD is NOT reset at each session. It runs over preloaded cross-session 1-minute history."
}

def _ema_old_ta_like(s, span):
    # We use standard EWM without session reset as a warm-up parity diagnostic.
    # This is intentionally not claimed as exact old 'ta' package parity.
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _macd(s, fast, slow):
    return _ema_old_ta_like(s, fast) - _ema_old_ta_like(s, slow)

def _prepare(path,label):
    x=pd.read_csv(path)
    miss=sorted(REQ-set(x.columns))
    if miss:
        raise ValueError(f"{label} missing columns: {miss}")
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

    daily=x.groupby(["symbol","date"],sort=True).agg(
        dclose=("close","last"),dvol=("volume","sum")
    ).reset_index()
    daily["prev_close"]=daily.groupby("symbol")["dclose"].shift(1)
    daily["prev_vol"]=daily.groupby("symbol")["dvol"].shift(1)
    x=x.merge(daily[["symbol","date","prev_close","prev_vol"]],
              on=["symbol","date"],how="left")

    # Continuous cross-session MACD diagnostic: crucial difference vs v29.
    x["macd12_26_cont"]=x.groupby("symbol",sort=False)["close"].transform(lambda s:_macd(s.astype(float),12,26))
    x["macd40_60_cont"]=x.groupby("symbol",sort=False)["close"].transform(lambda s:_macd(s.astype(float),40,60))
    return x

def _audit(path,label):
    x=_prepare(path,label)
    rows=[]

    for (sym,d),g0 in x.groupby(["symbol","date"],sort=False):
        g=g0.sort_values("timestamp").copy()
        if len(g)<10: continue
        pc=g["prev_close"].iloc[0]; pv=g["prev_vol"].iloc[0]
        if pd.isna(pc) or pd.isna(pv) or float(pc)<=0: continue

        first=float(g.iloc[0]["open"])
        price_ok=2<=first<=13
        dv_ok=float(pv)*first>500000

        first15=g[g["minute"]<585]
        if first15.empty: continue
        high15=float(first15["high"].max())
        cumvol=g["volume"].astype(float).cumsum()

        # Build session-reset MACD exactly like the v29 implementation for diagnosis.
        sess_close=g["close"].astype(float)
        sess_m1=_macd(sess_close,12,26)
        sess_m2=_macd(sess_close,40,60)

        base_candidates=0
        reset_m1_ok=0
        cont_m1_ok=0
        cont_both_ok=0
        first_base_time=None
        first_cont_time=None

        idxs=list(g.index)
        for j,idx in enumerate(idxs):
            r=g.loc[idx]
            minute=int(r["minute"])
            if not (585 < minute < 630):
                continue
            ch=(float(r["close"])-float(pc))/float(pc)
            if not (price_ok and dv_ok and ch>=0.035 and ch>0.04 and
                    float(r["close"])>high15 and float(cumvol.loc[idx])>30000):
                continue

            base_candidates+=1
            if first_base_time is None: first_base_time=str(r["timestamp"])

            # session-reset logic
            if j>=2 and pd.notna(sess_m1.iloc[j]):
                if float(sess_m1.iloc[j])>=0 and float(sess_m1.iloc[j-2])<float(sess_m1.iloc[j-1])<float(sess_m1.iloc[j]):
                    reset_m1_ok+=1

            # continuous cross-session logic
            pos=x.index.get_loc(idx)
            # safer: use symbol-frame positional neighborhood
            sg=x[x["symbol"].eq(sym)].sort_values("timestamp")
            locs=list(sg.index)
            k=locs.index(idx)
            cm1=sg.loc[idx,"macd12_26_cont"]
            cm2=sg.loc[idx,"macd40_60_cont"]
            m1ok=False; m2ok=False
            if k>=2 and pd.notna(cm1):
                a=float(sg.loc[locs[k-2],"macd12_26_cont"])
                b=float(sg.loc[locs[k-1],"macd12_26_cont"])
                c=float(cm1)
                m1ok=np.isfinite(a) and np.isfinite(b) and c>=0 and a<b<c
            if k>=1 and pd.notna(cm2):
                p=float(sg.loc[locs[k-1],"macd40_60_cont"])
                c=float(cm2)
                m2ok=np.isfinite(p) and c>=0 and c-p>=0

            if m1ok: cont_m1_ok+=1
            if m1ok and m2ok:
                cont_both_ok+=1
                if first_cont_time is None: first_cont_time=str(r["timestamp"])

        if base_candidates>0:
            rows.append({
                "symbol":sym,"date":str(d),
                "base_candidate_bars":base_candidates,
                "session_reset_macd12_26_pass_bars":reset_m1_ok,
                "continuous_macd12_26_pass_bars":cont_m1_ok,
                "continuous_both_macd_pass_bars":cont_both_ok,
                "first_base_candidate_time":first_base_time,
                "first_continuous_full_pass_time":first_cont_time
            })

    df=pd.DataFrame(rows)
    if df.empty:
        return df, {
            "label":label,"base_candidate_symbol_days":0,
            "session_reset_macd12_26_symbol_days":0,
            "continuous_macd12_26_symbol_days":0,
            "continuous_both_macd_symbol_days":0
        }

    summary={
        "label":label,
        "base_candidate_symbol_days":int(len(df)),
        "base_candidate_bars":int(df["base_candidate_bars"].sum()),
        "session_reset_macd12_26_symbol_days":int((df["session_reset_macd12_26_pass_bars"]>0).sum()),
        "continuous_macd12_26_symbol_days":int((df["continuous_macd12_26_pass_bars"]>0).sum()),
        "continuous_both_macd_symbol_days":int((df["continuous_both_macd_pass_bars"]>0).sum()),
        "continuous_both_macd_bars":int(df["continuous_both_macd_pass_bars"].sum()),
        "sample_continuous_passes":df[df["continuous_both_macd_pass_bars"]>0].head(20)[
            ["symbol","date","first_base_candidate_time","first_continuous_full_pass_time",
             "base_candidate_bars","continuous_macd12_26_pass_bars","continuous_both_macd_pass_bars"]
        ].to_dict("records")
    }
    return df,summary

def run_alpaca_macd_source_parity(progress_cb=None):
    if progress_cb:
        progress_cb(5,"Source-parity fact check: Alpaca preloads 1000 one-minute bars; MACD is cross-session")
    ddf,ds=_audit(DISCOVERY,"Discovery")
    if progress_cb:
        progress_cb(50,"True OOS cross-session MACD diagnostic")
    odf,os=_audit(TRUE_OOS,"True OOS")

    OUTDIR.mkdir(parents=True,exist_ok=True)
    ddf.to_csv(OUTDIR/"discovery_macd_parity_rows.csv",index=False)
    odf.to_csv(OUTDIR/"true_oos_macd_parity_rows.csv",index=False)

    cont_total=ds["continuous_both_macd_symbol_days"]+os["continuous_both_macd_symbol_days"]
    reset_total=ds["session_reset_macd12_26_symbol_days"]+os["session_reset_macd12_26_symbol_days"]

    if cont_total>0 and reset_total==0:
        verdict="SESSION_RESET_IMPLEMENTATION_BUG_CONFIRMED"
        next_step="FIX_29_MACD_WARMUP_ONLY_THEN_RERUN_SOURCE_LOCKED_RULES"
    elif cont_total==0:
        verdict="NO_PASS_EVEN_WITH_CONTINUOUS_5M_DIAGNOSTIC"
        next_step="DO_NOT_TUNE; exact old-ta/1-minute parity would be required before any further claim"
    else:
        verdict="MIXED"
        next_step="REVIEW_EXACT_OLD_TA_AND_1MIN_PARITY"

    result={
        "study":"Alpaca Momentum MACD Source-Parity Audit v1",
        "source_facts":SOURCE_FACTS,
        "strategy_changes":"NONE",
        "discovery":ds,
        "true_oos":os,
        "verdict":verdict,
        "important_limit":"The source uses 1000 PRELOADED 1-minute bars plus live 1-second aggregation. Our persistent historical replay is 5-minute. Therefore this audit can prove/reset-warmup mismatch, but it does NOT claim exact old-'ta' 1-minute numerical parity.",
        "next_step":next_step
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"MACD parity audit · {verdict}")
    return result
