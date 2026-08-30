import json, threading
from pathlib import Path
import pandas as pd
from engine import backtest

BASE=Path(__file__).resolve().parent
DATA=BASE/"daytrade_data.csv"
OUT=BASE/"outputs"
STATE={"running":False,"progress":0,"message":"尚未开始","results":[]}

def run_matrix():
    if STATE["running"]: return
    if not DATA.exists():
        STATE.update(running=False,progress=100,message="❌ 先下载 daytrade_data.csv")
        return
    def work():
        try:
            STATE.update(running=True,progress=1,message="读取数据…",results=[])
            df=pd.read_csv(DATA)
            cfg=json.loads((BASE/"config.json").read_text())
            combos=[(r,g,o) for r in (1.2,1.5,2.0,3.0)
                           for g in (.03,.04,.06) for o in (5,15,20)]
            rows=[]
            for i,(r,g,o) in enumerate(combos,1):
                c=dict(cfg)
                c["min_rvol"]=r
                c["min_gap_pct"]=g
                c["opening_range_minutes"]=o
                _,m=backtest(df,c)
                pf=m.get("profit_factor",0)
                pf=999.0 if pf=="inf" else float(pf)
                rows.append({
                    "rvol":r,"gap":g,"or_min":o,
                    "trades":m.get("trades",0),
                    "win_rate":m.get("win_rate",0),
                    "profit_factor":pf,
                    "avg_r":m.get("avg_r",0),
                    "max_drawdown_pct":m.get("max_drawdown_pct",0)
                })
                ranked=sorted(rows,key=lambda x:(x["profit_factor"],x["trades"]),reverse=True)
                STATE.update(progress=round(i/36*100),
                    message=f"正在跑 {i}/36：RVOL {r} / Gap {int(g*100)}% / OR {o}m",
                    results=ranked)
            rows=sorted(rows,key=lambda x:(x["profit_factor"],x["trades"]),reverse=True)
            OUT.mkdir(exist_ok=True)
            (OUT/"challenger_matrix.json").write_text(json.dumps(rows,indent=2))
            STATE.update(running=False,progress=100,
                message="✅ 36/36 完成。看稳定参数区域，不以单点最高 PF 下结论。",
                results=rows)
        except Exception as e:
            STATE.update(running=False,progress=100,message=f"❌ {type(e).__name__}: {e}")
    threading.Thread(target=work,daemon=True).start()
