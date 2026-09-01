
import os, json, hashlib, time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

DATA_ROOT = Path("/data/daytrade")
SIP_DATA = DATA_ROOT / "sip_data.csv"
ROOT = DATA_ROOT / "paper_forward_qc_v1"
LEDGER = ROOT / "paper_ledger.csv"
STATE = ROOT / "state.json"
FREEZE = ROOT / "freeze_manifest.json"
CACHE5 = ROOT / "cache_5m"
CACHE1 = ROOT / "cache_1m"

API_ROOT = "https://api.tiingo.com/tiingo/equity/intraday"

FROZEN_SPEC = {
    "strategy": "QuantConnect/Zarattini Stocks-in-Play ORB external benchmark",
    "frozen_at": "2026-09-01",
    "forward_start_date": "2026-08-31",
    "universe": "existing 125 Stocks-in-Play symbols from persistent sip_data.csv",
    "opening_range_minutes": 5,
    "atr_period_days": 14,
    "min_price": 5.0,
    "min_atr_dollars": 0.50,
    "min_relative_volume": 1.0,
    "max_positions": 20,
    "stop_atr_fraction": 0.10,
    "direction": "green first 5m -> long breakout; red first 5m -> short breakout",
    "entry": "stop order at first 5m high/low, active from next 1m bar",
    "exit": "0.1*ATR stop or EOD",
    "execution": "1-minute stop-order semantics; gap-through fills at minute open, otherwise stop price",
    "tuning": "FORBIDDEN during forward test",
}

def _spec_hash():
    raw = json.dumps(FROZEN_SPEC, sort_keys=True, separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()

def _ensure_dirs():
    for p in [ROOT,CACHE5,CACHE1]:
        p.mkdir(parents=True, exist_ok=True)
    if not FREEZE.exists():
        FREEZE.write_text(json.dumps({
            "spec": FROZEN_SPEC,
            "spec_sha256": _spec_hash(),
            "created_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat()
        }, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed_dates": [], "failed_dates": []}

def _save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)

def _last_completed_ny_date():
    ny = datetime.now(tz=ZoneInfo("America/New_York"))
    # Only regard today as complete after 16:10 NY.
    if (ny.hour, ny.minute) >= (16,10) and ny.weekday() < 5:
        return ny.date()
    d = ny.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

def _symbols():
    if not SIP_DATA.exists():
        raise FileNotFoundError("Persistent sip_data.csv missing")
    x = pd.read_csv(SIP_DATA, usecols=["symbol"])
    syms = sorted(x["symbol"].dropna().astype(str).unique().tolist())
    if len(syms) != 125:
        raise ValueError(f"Expected 125 symbols, found {len(syms)}")
    return syms

def _download(symbol, date_str, freq, token, sess):
    root = CACHE5 if freq=="5min" else CACHE1
    p = root / symbol / f"{date_str}.csv"
    if p.exists():
        return p, "CACHE"
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API_ROOT}/{symbol}/prices"
    params = {
        "startDate": date_str, "endDate": date_str, "resampleFreq": freq,
        "columns": "open,high,low,close,volume", "token": token
    }
    last = None
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(1.5*(attempt+1)); continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                raise ValueError("empty response")
            df = pd.DataFrame(data)
            if "date" not in df.columns:
                raise ValueError(f"missing date column {list(df.columns)}")
            ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
            df["timestamp"] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["timestamp","open","high","low","close"]).copy()
            df["time"] = df["timestamp"].dt.strftime("%H:%M")
            df = df[(df["time"] >= "09:30") & (df["time"] <= "15:59")].copy()
            if df.empty:
                raise ValueError("no regular-session rows")
            df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp")
            tmp = p.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(p)
            return p, "DOWNLOADED"
        except Exception as e:
            last=e
            time.sleep(1.0*(attempt+1))
    raise RuntimeError(f"{symbol} {date_str} {freq}: {last}")

def _history_features(target_date):
    x = pd.read_csv(SIP_DATA)
    need={"symbol","timestamp","open","high","low","close","volume"}
    miss=sorted(need-set(x.columns))
    if miss: raise ValueError(f"sip_data missing {miss}")
    ts=pd.to_datetime(x["timestamp"],errors="coerce",utc=True)
    x["timestamp"]=ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    x=x.dropna(subset=["timestamp"]).copy()
    x["date"]=x["timestamp"].dt.date
    x["time"]=x["timestamp"].dt.strftime("%H:%M")
    x=x[x["date"] < target_date].copy()
    x=x[(x["time"]>="09:30")&(x["time"]<="15:55")].copy()
    x.sort_values(["symbol","timestamp"],inplace=True)

    daily=x.groupby(["symbol","date"],sort=True).agg(
        dhigh=("high","max"),dlow=("low","min"),dclose=("close","last")
    ).reset_index()
    daily["prev_close"]=daily.groupby("symbol")["dclose"].shift(1)
    daily["tr"]=np.maximum(
        daily["dhigh"]-daily["dlow"],
        np.maximum((daily["dhigh"]-daily["prev_close"]).abs(),
                   (daily["dlow"]-daily["prev_close"]).abs())
    )
    atr={}
    for sym,g in daily.groupby("symbol"):
        vals=g["tr"].dropna().tail(14)
        if len(vals)==14: atr[str(sym)]=float(vals.mean())

    first=x.groupby(["symbol","date"],sort=True).first().reset_index()
    ov={}
    for sym,g in first.groupby("symbol"):
        vals=pd.to_numeric(g["volume"],errors="coerce").dropna().tail(14)
        if len(vals)==14 and float(vals.mean())>0: ov[str(sym)]=float(vals.mean())
    return atr,ov

def _simulate_1m(side, entry, stop, bars):
    bars=bars.copy()
    bars["timestamp"]=pd.to_datetime(bars["timestamp"],errors="coerce")
    bars=bars.dropna(subset=["timestamp"]).sort_values("timestamp")
    direction=1 if side=="LONG" else -1
    hit_idx=None; fill=None; hit_time=None
    for idx,b in bars.iterrows():
        if side=="LONG":
            touched=float(b["high"])>=entry
            if touched:
                fill=max(float(b["open"]),entry)
        else:
            touched=float(b["low"])<=entry
            if touched:
                fill=min(float(b["open"]),entry)
        if touched:
            hit_idx=idx; hit_time=b["timestamp"]; break
    if hit_idx is None:
        return None

    risk=abs(entry-stop)
    later=bars.loc[bars.index>=hit_idx]
    exit_px=None; reason="EOD"; exit_time=None
    # stop becomes active immediately after fill; same 1m bar is still ambiguous,
    # so conservatively activate from next minute for forward bookkeeping.
    later=later.loc[later.index>hit_idx]
    for _,b in later.iterrows():
        stopped=(float(b["low"])<=stop) if side=="LONG" else (float(b["high"])>=stop)
        if stopped:
            exit_px=stop; reason="STOP"; exit_time=b["timestamp"]; break
    if exit_px is None:
        exit_px=float(bars.iloc[-1]["close"]); exit_time=bars.iloc[-1]["timestamp"]
    pnl_r=(exit_px-fill)*direction/risk
    return {
        "entry_fill":float(fill),"entry_time":str(hit_time),
        "exit_fill":float(exit_px),"exit_time":str(exit_time),
        "exit_reason":reason,"pnl_r":float(pnl_r)
    }

def _summary(ledger):
    if ledger.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"median_r":0.0,"max_dd_r":0.0}
    r=pd.to_numeric(ledger["pnl_r"],errors="coerce").dropna()
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    eq=r.cumsum(); dd=eq-eq.cummax()
    return {
        "trades":int(len(r)),
        "win_rate":round(float((r>0).mean()),4),
        "profit_factor":round(float(pf),4),
        "avg_r":round(float(r.mean()),4),
        "median_r":round(float(r.median()),4),
        "max_dd_r":round(float(dd.min()),4)
    }

def run_paper_forward_update(progress_cb=None):
    _ensure_dirs()
    token=os.getenv("TIINGO_TOKEN")
    if not token: raise RuntimeError("TIINGO_TOKEN missing")

    target=_last_completed_ny_date()
    start=pd.Timestamp(FROZEN_SPEC["forward_start_date"]).date()
    if target < start:
        return {"study":"Paper Forward Test v1","verdict":"WAIT","message":f"No completed forward date yet. First eligible date {start}."}

    st=_load_state()
    processed=set(st.get("processed_dates",[]))
    # Process every unprocessed weekday from start through latest completed date.
    dates=[]
    d=start
    while d<=target:
        if d.weekday()<5 and str(d) not in processed:
            dates.append(d)
        d += timedelta(days=1)

    if not dates:
        ledger=pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
        return {
            "study":"Paper Forward Test v1","verdict":"UP_TO_DATE",
            "frozen_spec_sha256":_spec_hash(),
            "processed_dates":sorted(processed),
            "summary":_summary(ledger),
            "ledger_rows":int(len(ledger)),
            "next_step":"WAIT_FOR_NEW_COMPLETED_US_TRADING_DAY"
        }

    syms=_symbols()
    sess=requests.Session()
    all_new=[]

    for di,target_date in enumerate(dates,1):
        date_str=str(target_date)
        if progress_cb:
            progress_cb(2, f"{date_str}: build prior-14 ATR/open-volume context")
        atr,ov=_history_features(target_date)

        first_rows=[]
        errors=[]
        for i,sym in enumerate(syms,1):
            try:
                p,_=_download(sym,date_str,"5min",token,sess)
                x=pd.read_csv(p)
                x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
                x=x.dropna(subset=["timestamp"]).sort_values("timestamp")
                if x.empty: continue
                b=x.iloc[0]
                if sym not in atr or sym not in ov: continue
                rvol=float(b["volume"])/ov[sym] if ov[sym]>0 else np.nan
                first_rows.append({
                    "symbol":sym,"date":date_str,"open":float(b["open"]),
                    "high":float(b["high"]),"low":float(b["low"]),"close":float(b["close"]),
                    "volume":float(b["volume"]),"atr14":atr[sym],"rvol":rvol,
                    "first_ts":str(b["timestamp"])
                })
            except Exception as e:
                errors.append(f"{sym}: {e}")
            if progress_cb and (i%20==0 or i==len(syms)):
                base=5+int(35*i/len(syms))
                progress_cb(base, f"{date_str}: 5min {i}/{len(syms)} · errors {len(errors)}")

        first=pd.DataFrame(first_rows)
        if first.empty:
            st.setdefault("failed_dates",[]).append({"date":date_str,"error":"no first bars"})
            _save_state(st)
            continue

        cand=first[
            (first["close"]>FROZEN_SPEC["min_price"]) &
            (first["atr14"]>FROZEN_SPEC["min_atr_dollars"]) &
            (first["rvol"]>FROZEN_SPEC["min_relative_volume"])
        ].copy()
        cand=cand.sort_values("rvol",ascending=False).head(FROZEN_SPEC["max_positions"])

        day_rows=[]
        for j,r in enumerate(cand.to_dict("records"),1):
            if r["close"]>r["open"]:
                side="LONG"; entry=float(r["high"]); stop=entry-FROZEN_SPEC["stop_atr_fraction"]*float(r["atr14"])
            elif r["close"]<r["open"]:
                side="SHORT"; entry=float(r["low"]); stop=entry+FROZEN_SPEC["stop_atr_fraction"]*float(r["atr14"])
            else:
                continue
            try:
                p,_=_download(r["symbol"],date_str,"1min",token,sess)
                bars=pd.read_csv(p)
                # only after first 5m opening range is known
                bars["timestamp"]=pd.to_datetime(bars["timestamp"],errors="coerce")
                bars=bars[bars["timestamp"]>=pd.Timestamp(f"{date_str} 09:35:00")].copy()
                sim=_simulate_1m(side,entry,stop,bars)
                if sim is None:
                    continue
                day_rows.append({
                    "date":date_str,"symbol":r["symbol"],"side":side,
                    "rvol":round(float(r["rvol"]),4),"atr14":round(float(r["atr14"]),4),
                    "intended_entry":round(entry,4),"stop":round(stop,4),
                    **sim
                })
            except Exception as e:
                errors.append(f"{r['symbol']} 1m: {e}")
            if progress_cb:
                progress_cb(45+int(40*j/max(len(cand),1)), f"{date_str}: 1min execution {j}/{len(cand)}")

        if day_rows:
            all_new.extend(day_rows)

        processed.add(date_str)
        st["processed_dates"]=sorted(processed)
        if errors:
            st.setdefault("date_errors",{})[date_str]=errors[-50:]
        _save_state(st)

    newdf=pd.DataFrame(all_new)
    if LEDGER.exists():
        old=pd.read_csv(LEDGER)
        ledger=pd.concat([old,newdf],ignore_index=True) if not newdf.empty else old
    else:
        ledger=newdf
    if not ledger.empty:
        ledger=ledger.drop_duplicates(["date","symbol","side","entry_time"],keep="last").sort_values(["date","entry_time","symbol"])
        tmp=LEDGER.with_suffix(".tmp")
        ledger.to_csv(tmp,index=False); tmp.replace(LEDGER)
    else:
        ledger=pd.DataFrame()

    result={
        "study":"Paper Forward Test v1",
        "mode":"FROZEN_FORWARD_ONLY",
        "frozen_spec_sha256":_spec_hash(),
        "forward_start_date":FROZEN_SPEC["forward_start_date"],
        "processed_dates":sorted(processed),
        "new_dates_processed":[str(x) for x in dates],
        "new_trades":int(len(newdf)),
        "ledger_rows":int(len(ledger)),
        "summary":_summary(ledger),
        "guardrail":"Do not tune strategy during forward test. Only append newly completed US trading days.",
        "paper_gate":"Need at least 20 completed forward trading days AND >=100 trades before judging live readiness."
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"latest_result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Paper Forward updated · dates {len(dates)} · new trades {len(newdf)}")
    return result
