
import json
from pathlib import Path
import numpy as np
import pandas as pd
from engine import prepare

SETUPS=["ORB_BREAKOUT","OPENING_DRIVE","VWAP_RECLAIM","FIRST_PULLBACK"]

def _metrics(df):
    if df.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"stop_rate":0.0}
    r=pd.to_numeric(df["pnl_r"],errors="coerce").fillna(0.0)
    gp=float(r[r>0].sum()); gl=float(-r[r<0].sum())
    pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
    return {"trades":int(len(df)),"win_rate":round(float((r>0).mean()),4),"profit_factor":round(pf,4),
            "avg_r":round(float(r.mean()),4),"stop_rate":round(float(df["exit_reason"].astype(str).eq("STOP").mean()),4)}

def _prep(path):
    x=prepare(pd.read_csv(path)).copy()
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
    x=x.dropna(subset=["timestamp"]); x["date"]=x["timestamp"].dt.strftime("%Y-%m-%d")
    x=x[(x["time"]>="09:30")&(x["time"]<"16:00")].copy()
    x.sort_values(["symbol","timestamp"],inplace=True)
    groups={}
    for (sym,dt),g in x.groupby(["symbol","date"],sort=True):
        g=g.copy().reset_index(drop=True)
        typ=(g["high"]+g["low"]+g["close"])/3.0
        g["vwap_t"]=(typ*g["volume"]).cumsum()/g["volume"].cumsum().replace(0,np.nan)
        groups[(str(sym),str(dt))]=g
    last=x.groupby(["symbol","date"],sort=True)["close"].last().reset_index(name="session_close")
    last["prev_close"]=last.groupby("symbol")["session_close"].shift(1)
    prev={(str(r.symbol),str(r.date)):float(r.prev_close) for r in last.itertuples(index=False)
          if pd.notna(r.prev_close) and float(r.prev_close)>0}
    return groups,prev

def _exit(reg,ets,entry,stop,target,cfg):
    post=reg[reg["timestamp"]>=ets]
    if post.empty:return None
    for b in post.itertuples(index=False):
        if float(b.low)<=stop:return stop-cfg["slippage_per_share"],b.timestamp,"STOP"
        if float(b.high)>=target:return target-cfg["slippage_per_share"],b.timestamp,"TARGET"
        if b.time>=cfg["eod_exit"]:return float(b.close)-cfg["slippage_per_share"],b.timestamp,"EOD"
    z=post.iloc[-1]
    return float(z["close"])-cfg["slippage_per_share"],z["timestamp"],"LAST_BAR"

def _next(reg,ts):
    x=reg[reg["timestamp"]>ts]
    return None if x.empty else x.iloc[0]

def _orb(reg,prev,cfg):
    if float(reg.iloc[0]["open"])/prev-1<0.03:return None
    o=reg[reg["time"]<"09:50"]
    if o.empty:return None
    hi,lo=float(o["high"].max()),float(o["low"].min())
    c=reg[(reg["time"]>="09:50")&(reg["time"]<="11:00")]
    c=c[(c["high"]>hi)&(pd.to_numeric(c["rvol_bar"],errors="coerce")>=3)&(c["close"]>c["vwap_t"])]
    if c.empty:return None
    e=_next(reg,c.iloc[0]["timestamp"])
    if e is None:return None
    entry=float(e["open"])+cfg["slippage_per_share"]; stop=max(lo,entry-0.5*max(hi-lo,0.01)); target=entry+2*max(entry-stop,0.01)
    return e,stop,target

def _drive(reg,prev,cfg):
    if float(reg.iloc[0]["open"])/prev-1<0.02:return None
    f=reg[reg["time"]<"09:45"]
    if len(f)<2:return None
    if float(f.iloc[-1]["close"])/float(reg.iloc[0]["open"])-1<0.01:return None
    if float(pd.to_numeric(f["rvol_bar"],errors="coerce").median())<2:return None
    hi,lo=float(f["high"].max()),float(f["low"].min())
    c=reg[(reg["time"]>="09:45")&(reg["time"]<="10:30")]
    c=c[(c["high"]>hi)&(c["close"]>c["vwap_t"])]
    if c.empty:return None
    e=_next(reg,c.iloc[0]["timestamp"])
    if e is None:return None
    entry=float(e["open"])+cfg["slippage_per_share"]; stop=max(lo,entry*0.985); target=entry+2*max(entry-stop,0.01)
    return e,stop,target

def _reclaim(reg,prev,cfg):
    if float(reg.iloc[0]["open"])/prev-1<0.02:return None
    w=reg[(reg["time"]>="09:45")&(reg["time"]<="11:30")].copy()
    if len(w)<3:return None
    close=pd.to_numeric(w["close"],errors="coerce"); vw=pd.to_numeric(w["vwap_t"],errors="coerce")
    c=w[(close>vw)&(close.shift(1)<vw.shift(1))]
    if c.empty:return None
    s=c.iloc[0]; e=_next(reg,s["timestamp"])
    if e is None:return None
    entry=float(e["open"])+cfg["slippage_per_share"]; stop=min(float(s["low"]),entry*0.985); target=entry+2*max(entry-stop,0.01)
    return e,stop,target

def _pullback(reg,prev,cfg):
    if float(reg.iloc[0]["open"])/prev-1<0.03:return None
    f=reg[reg["time"]<"09:50"]
    if f.empty:return None
    hi=float(f["high"].max())
    br=reg[(reg["time"]>="09:50")&(reg["time"]<="10:45")]
    br=br[(br["high"]>hi)&(br["close"]>br["vwap_t"])]
    if br.empty:return None
    a=reg[reg["timestamp"]>br.iloc[0]["timestamp"]]
    pb=a[(a["close"]<a["open"])&(a["low"]>a["vwap_t"])]
    if pb.empty:return None
    p=pb.iloc[0]; e=_next(reg,p["timestamp"])
    if e is None or float(e["high"])<=float(p["high"]):return None
    e=e.copy(); e["open"]=max(float(e["open"]),float(p["high"]))
    entry=float(e["open"])+cfg["slippage_per_share"]; stop=float(p["low"]); target=entry+2*max(entry-stop,0.01)
    return e,stop,target

FUNCS={"ORB_BREAKOUT":_orb,"OPENING_DRIVE":_drive,"VWAP_RECLAIM":_reclaim,"FIRST_PULLBACK":_pullback}

def _run(groups,prev,cfg,setup):
    eq=float(cfg["starting_equity"]); rows=[]; fn=FUNCS[setup]
    for key in sorted(groups):
        reg=groups[key]; pv=prev.get(key)
        if pv is None or len(reg)<20:continue
        sig=fn(reg,pv,cfg)
        if sig is None:continue
        e,stop,target=sig
        entry=float(e["open"])+cfg["slippage_per_share"]; risk=max(entry-stop,0.01)
        sh=int((eq*cfg["risk_per_trade_pct"])//risk)
        if sh<=0:continue
        out=_exit(reg,e["timestamp"],entry,stop,target,cfg)
        if not out:continue
        xp,xt,reason=out
        pnl=(xp-entry)*sh-cfg["commission_per_share"]*sh*2
        pr=pnl/(risk*sh); eq+=pnl
        rows.append({"setup":setup,"symbol":key[0],"date":key[1],"pnl":round(pnl,2),"pnl_r":round(pr,4),"exit_reason":reason})
    df=pd.DataFrame(rows)
    return df,_metrics(df)

def run_setup_tournament(discovery_path,oos_path,config_path,outdir,progress_cb=None):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if progress_cb:progress_cb(5,"读取 Discovery Year")
    dg,dp=_prep(discovery_path)
    if progress_cb:progress_cb(35,"读取 True OOS Year")
    og,op=_prep(oos_path)
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); results=[]
    for i,s in enumerate(SETUPS,1):
        if progress_cb:progress_cb(40+i*12,f"测试 {s}")
        dt,dm=_run(dg,dp,cfg,s); ot,om=_run(og,op,cfg,s)
        dt.to_csv(out/f"{s.lower()}_discovery.csv",index=False); ot.to_csv(out/f"{s.lower()}_oos.csv",index=False)
        score=(1 if dm["profit_factor"]>1 else 0)+(1 if dm["avg_r"]>0 else 0)+(2 if om["profit_factor"]>1 else 0)+(2 if om["avg_r"]>0 else 0)+(1 if om["trades"]>=40 else 0)
        verdict="KEEP" if score>=5 else ("WATCH" if score>=3 else "DROP")
        results.append({"setup":s,"discovery":dm,"oos":om,"score":score,"verdict":verdict})
    results.sort(key=lambda x:(x["verdict"]=="KEEP",x["oos"]["profit_factor"],x["oos"]["avg_r"],x["oos"]["trades"]),reverse=True)
    result={"study":"US DayTrade Setup Tournament v1","ranking_rule":"True OOS first; no threshold sweep.","results":results,"winner":results[0] if results else None}
    (out/"setup_tournament_v1.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if progress_cb:progress_cb(100,"Setup Tournament v1 完成")
    return result
