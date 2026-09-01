
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("/data/daytrade")
DISCOVERY = DATA_ROOT / "sip_data.csv"
TRUE_OOS = DATA_ROOT / "oos_v08" / "sip_oos_2024-08-30_2025-08-29.csv"
OUTDIR = DATA_ROOT / "alpaca_momentum_filter_funnel_v1"

REQ = {"symbol","timestamp","open","high","low","close","volume"}

SPEC = {
    "min_share_price": 2.0,
    "max_share_price": 13.0,
    "min_prev_day_dollar_volume": 500000.0,
    "universe_change_filter_pct": 3.5,
    "entry_change_filter_pct": 4.0,
    "entry_window": "09:45-10:30",
    "min_intraday_volume": 30000.0,
    "macd_1": "MACD(12,26) >=0 and rising 3 bars",
    "macd_2": "MACD(40,60) >=0 and slope >=0"
}

def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _macd(close, fast, slow):
    return _ema(close,fast)-_ema(close,slow)

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
        dclose=("close","last"), dvol=("volume","sum")
    ).reset_index()
    daily["prev_close"]=daily.groupby("symbol")["dclose"].shift(1)
    daily["prev_vol"]=daily.groupby("symbol")["dvol"].shift(1)
    x=x.merge(daily[["symbol","date","prev_close","prev_vol"]],
              on=["symbol","date"],how="left")
    return x

def _audit_period(path,label):
    x=_prepare(path,label)
    records=[]

    # symbol-day level funnel
    for (sym,d),g0 in x.groupby(["symbol","date"],sort=False):
        g=g0.sort_values("timestamp").reset_index(drop=True)
        if len(g)<15:
            continue

        first=float(g.iloc[0]["open"])
        pc=g["prev_close"].iloc[0]
        pv=g["prev_vol"].iloc[0]

        f_price = SPEC["min_share_price"] <= first <= SPEC["max_share_price"]
        f_prev_dv = bool(pd.notna(pv) and float(pv)*first > SPEC["min_prev_day_dollar_volume"])

        if pd.isna(pc) or float(pc)<=0:
            max_change=-999
        else:
            max_change=float(((g["close"].astype(float)-float(pc))/float(pc)).max())

        f_change35 = max_change >= 0.035

        # breakout can only be judged if there is a proper first-15m high
        first15=g[g["minute"]<585]
        high15=float(first15["high"].max()) if not first15.empty else np.nan
        in_window=g[(g["minute"]>585)&(g["minute"]<630)].copy()
        f_breakout=False
        f_change4=False
        f_cumvol=False
        f_macd1=False
        f_macd2=False
        final=False

        if not in_window.empty and pd.notna(pc) and float(pc)>0 and np.isfinite(high15):
            close=g["close"].astype(float)
            m1=_macd(close,12,26)
            m2=_macd(close,40,60)
            cumvol=g["volume"].astype(float).cumsum()

            for i in in_window.index:
                r=g.loc[i]
                ch=(float(r["close"])-float(pc))/float(pc)
                if ch>0.04:
                    f_change4=True
                if float(r["close"])>high15:
                    f_breakout=True
                if float(cumvol.loc[i])>SPEC["min_intraday_volume"]:
                    f_cumvol=True
                if i>=2 and pd.notna(m1.loc[i]):
                    if float(m1.loc[i])>=0 and float(m1.loc[i-2])<float(m1.loc[i-1])<float(m1.loc[i]):
                        f_macd1=True
                if i>=1 and pd.notna(m2.loc[i]) and pd.notna(m2.loc[i-1]):
                    if float(m2.loc[i])>=0 and float(m2.loc[i])-float(m2.loc[i-1])>=0:
                        f_macd2=True

                if (
                    f_price and f_prev_dv and ch>=0.035 and ch>0.04 and
                    float(r["close"])>high15 and float(cumvol.loc[i])>SPEC["min_intraday_volume"] and
                    i>=2 and pd.notna(m1.loc[i]) and
                    float(m1.loc[i])>=0 and float(m1.loc[i-2])<float(m1.loc[i-1])<float(m1.loc[i]) and
                    i>=1 and pd.notna(m2.loc[i]) and pd.notna(m2.loc[i-1]) and
                    float(m2.loc[i])>=0 and float(m2.loc[i])-float(m2.loc[i-1])>=0
                ):
                    final=True
                    break

        records.append({
            "symbol":sym,"date":str(d),"first_price":first,
            "price_2_13":f_price,
            "prev_dollar_volume_500k":f_prev_dv,
            "change_ge_3_5pct":f_change35,
            "change_gt_4pct_in_window":f_change4,
            "breakout_first15":f_breakout,
            "cumvol_gt_30k":f_cumvol,
            "macd12_26_ok":f_macd1,
            "macd40_60_ok":f_macd2,
            "final_entry_eligible":final
        })

    df=pd.DataFrame(records)
    if df.empty:
        return df, {}

    # sequential funnel, so each stage means "passed all previous stages too"
    stages=[
        ("All symbol-days", np.ones(len(df),dtype=bool)),
        ("Price $2-$13", df["price_2_13"]),
        ("+ Prev dollar volume >$500k", df["price_2_13"] & df["prev_dollar_volume_500k"]),
        ("+ Change >=3.5%", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"]),
        ("+ Change >4% in entry window", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"] & df["change_gt_4pct_in_window"]),
        ("+ Breakout first-15m high", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"] & df["change_gt_4pct_in_window"] & df["breakout_first15"]),
        ("+ Cum volume >30k", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"] & df["change_gt_4pct_in_window"] & df["breakout_first15"] & df["cumvol_gt_30k"]),
        ("+ MACD(12,26)", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"] & df["change_gt_4pct_in_window"] & df["breakout_first15"] & df["cumvol_gt_30k"] & df["macd12_26_ok"]),
        ("+ MACD(40,60)", df["price_2_13"] & df["prev_dollar_volume_500k"] & df["change_ge_3_5pct"] & df["change_gt_4pct_in_window"] & df["breakout_first15"] & df["cumvol_gt_30k"] & df["macd12_26_ok"] & df["macd40_60_ok"]),
        ("Final eligible", df["final_entry_eligible"])
    ]
    funnel=[]
    prev=None
    for name,mask in stages:
        count=int(pd.Series(mask).sum())
        funnel.append({
            "stage":name,
            "count":count,
            "retention_from_start":round(count/len(df),4),
            "retention_from_prev":round(count/prev,4) if prev not in (None,0) else (1.0 if prev is None else 0.0)
        })
        prev=count

    # Universe diagnosis
    unique_syms=int(df["symbol"].nunique())
    price_syms=int(df.loc[df["price_2_13"],"symbol"].nunique())
    final_syms=int(df.loc[df["final_entry_eligible"],"symbol"].nunique())

    summary={
        "label":label,
        "symbol_days":int(len(df)),
        "symbols":unique_syms,
        "symbols_ever_price_2_13":price_syms,
        "symbols_ever_final_eligible":final_syms,
        "funnel":funnel,
        "top_price_compatible_symbols":(
            df[df["price_2_13"]]["symbol"].value_counts().head(20).to_dict()
        )
    }
    return df,summary

def run_alpaca_filter_funnel_audit(progress_cb=None):
    if progress_cb:
        progress_cb(5,"Discovery filter funnel")
    ddf,ds=_audit_period(DISCOVERY,"Discovery")
    if progress_cb:
        progress_cb(50,"True OOS filter funnel")
    odf,os=_audit_period(TRUE_OOS,"True OOS")

    OUTDIR.mkdir(parents=True,exist_ok=True)
    ddf.to_csv(OUTDIR/"discovery_symbol_day_filters.csv",index=False)
    odf.to_csv(OUTDIR/"true_oos_symbol_day_filters.csv",index=False)

    # Decide whether zero trades are caused by incompatible universe vs implementation.
    def final_count(s):
        return next((x["count"] for x in s["funnel"] if x["stage"]=="Final eligible"),0)
    def price_count(s):
        return next((x["count"] for x in s["funnel"] if x["stage"]=="Price $2-$13"),0)

    d_price=price_count(ds); o_price=price_count(os)
    d_final=final_count(ds); o_final=final_count(os)

    if d_price==0 and o_price==0:
        verdict="INCOMPATIBLE_UNIVERSE_PRICE"
    elif d_final==0 and o_final==0:
        verdict="FILTERS_ELIMINATE_ALL"
    else:
        verdict="IMPLEMENTATION_REVIEW_REQUIRED"

    result={
        "study":"Alpaca Momentum Filter Funnel Audit v1",
        "strategy_changes":"NONE",
        "spec":SPEC,
        "discovery":ds,
        "true_oos":os,
        "verdict":verdict,
        "interpretation":{
            "INCOMPATIBLE_UNIVERSE_PRICE":"The 125-stock pool contains no symbol-days in the source strategy's $2-$13 price universe. 29 should be marked universe-incompatible, not strategy-failed.",
            "FILTERS_ELIMINATE_ALL":"Some symbol-days pass the price universe, but later source filters eliminate every setup. 29 remains unproven on this pool and should not be rescued by changing thresholds.",
            "IMPLEMENTATION_REVIEW_REQUIRED":"At least some symbol-days appear final-eligible, so 29 returning 0 trades indicates an implementation mismatch that should be fixed before judging the strategy."
        }[verdict],
        "next_step":{
            "INCOMPATIBLE_UNIVERSE_PRICE":"DO_NOT_TUNE_29; choose a challenger designed for the current 125-stock universe or build a separate source-faithful low-price universe later.",
            "FILTERS_ELIMINATE_ALL":"DO_NOT_TUNE_29; mark as incompatible/unsupported on this pool.",
            "IMPLEMENTATION_REVIEW_REQUIRED":"FIX_29_EXECUTION_ONLY; do not change source parameters."
        }[verdict]
    }
    (OUTDIR/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Funnel Audit complete · {verdict}")
    return result
