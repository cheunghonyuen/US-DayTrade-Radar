import json, os, threading, time
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from engine import backtest

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
DATA = BASE / "daytrade_data.csv"
OUT = BASE / "outputs"
LOCK = threading.Lock()

JOBS = {
    "download": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "backtest": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "matrix": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "results": []},
}

HTML = """
<!doctype html>
<html lang="zh">
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>US DayTrade Radar v0.3</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;background:#f4f6f8;margin:0;padding:16px;color:#111}
.wrap{max-width:900px;margin:auto}
.card{background:#fff;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 1px 7px rgba(0,0,0,.08)}
h1{font-size:28px;margin:4px 0 18px}
h2{font-size:24px}
.muted{color:#667085}
.ok{color:#0a8f42}.bad{color:#c62828}
button{background:#111827;color:#fff;border:0;border-radius:10px;padding:12px 16px;font-size:16px;margin:6px 6px 6px 0}
button:disabled{opacity:.45}
input{padding:10px;border:1px solid #ccd2da;border-radius:8px;font-size:16px;width:150px}
pre{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;padding:12px;border-radius:10px;font-size:13px}
.row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:10px 0 7px}
.track{height:16px;border-radius:999px;background:#e5e7eb;overflow:hidden}
.bar{height:100%;width:0;background:#111827;transition:width .3s ease}
.bar.done{background:#0a8f42}.bar.fail{background:#c62828}
.msg{font-size:13px;color:#667085;margin-top:7px;white-space:pre-wrap;word-break:break-word}
.metrics{display:flex;flex-wrap:wrap;margin-top:12px}
.metric{display:inline-block;min-width:120px;margin:5px;padding:10px;background:#f7f8fa;border-radius:10px}
.scroll{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:700px}
th,td{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right}
th:first-child,td:first-child{text-align:left}
.pass{font-weight:700}
</style>
</head>
<body>
<div class="wrap">
<h1>🇺🇸 US DayTrade Radar <small>v0.3</small></h1>

<div class="card">
<h2>1. Preflight</h2>
<p class="muted">先检查文件、依赖、Tiingo Token 和接口；失败就不允许长跑。</p>
<form method="post" action="/preflight"><button>Run Preflight</button></form>
{% if preflight %}
<pre class="{{ 'ok' if preflight.ok else 'bad' }}">{{ preflight.text }}</pre>
{% endif %}
</div>

<div class="card">
<h2>2. 下载 Tiingo 样本</h2>
<label>Start <input id="start" type="date" value="{{ start }}"></label>
<label>End <input id="end" type="date" value="{{ end }}"></label><br>
<button id="downloadBtn" type="button">Download 20 Stocks / 5min</button>
<div class="row"><b>下载进度</b><span id="downloadPct">0%</span></div>
<div class="track"><div id="downloadBar" class="bar"></div></div>
<div id="downloadMsg" class="msg">尚未开始</div>
</div>

<div class="card">
<h2>3. ORB Backtest</h2>
<button id="backtestBtn" type="button">Run Baseline</button>
<div class="row"><b>回测进度</b><span id="backtestPct">0%</span></div>
<div class="track"><div id="backtestBar" class="bar"></div></div>
<div id="backtestMsg" class="msg">尚未开始</div>
<div id="metricsBox" class="metrics"></div>
</div>

<div class="card">
<h2>4. Challenger Matrix</h2>
<p><b>36组：</b> RVOL 1.2 / 1.5 / 2.0 / 3.0 × Gap ≥3% / 4% / 6% × OR 5 / 15 / 20min</p>
<p class="muted">固定其他规则不动。自动按 PF 排名，同时显示 Trades / Win Rate / Avg R / MaxDD。目的是找稳定参数区域，不是挑单一最高 PF。</p>
<button id="matrixBtn" type="button">Run 36 Challengers</button>
<div class="row"><b>Matrix 进度</b><span id="matrixPct">0%</span></div>
<div class="track"><div id="matrixBar" class="bar"></div></div>
<div id="matrixMsg" class="msg">尚未开始</div>
<div id="matrixResults" class="scroll"></div>
</div>

<div class="card">
<h3>状态</h3>
<p>Data: <span id="dataState">{{ "✅ Data ready" if data_exists else "❌ No data" }}</span></p>
<p class="muted">Research / Paper only. No IBKR live orders.</p>
</div>
</div>

<script>
const $ = (id)=>document.getElementById(id);

function setProgress(kind, data){
  const bar=$(kind+"Bar"), pct=$(kind+"Pct"), msg=$(kind+"Msg"), btn=$(kind+"Btn");
  const p=Math.max(0,Math.min(100,Number(data.progress||0)));
  bar.style.width=p+"%";
  pct.textContent=p+"%";
  msg.textContent=data.message||"";
  btn.disabled=!!data.running;
  bar.classList.remove("done","fail");
  if(!data.running && data.ok===true) bar.classList.add("done");
  if(!data.running && data.ok===false) bar.classList.add("fail");
}

function renderMetrics(metrics){
  if(!metrics) return;
  const box=$("metricsBox");
  box.innerHTML="";
  Object.entries(metrics).forEach(([k,v])=>{
    const d=document.createElement("div");
    d.className="metric";
    d.innerHTML="<b>"+k+"</b><br>"+v;
    box.appendChild(d);
  });
}

function renderMatrix(rows){
  if(!rows || !rows.length) return;
  let h='<table><thead><tr><th>#</th><th>RVOL</th><th>Gap</th><th>OR</th><th>Trades</th><th>Win%</th><th>PF</th><th>Avg R</th><th>MaxDD%</th></tr></thead><tbody>';
  rows.slice(0,15).forEach((r,i)=>{
    const cls=(r.profit_factor>=1.2 && r.trades>=30) ? "pass" : "";
    h+=`<tr class="${cls}"><td>${i+1}</td><td>${r.rvol}</td><td>${Math.round(r.gap*100)}%</td><td>${r.or_min}m</td><td>${r.trades}</td><td>${(Number(r.win_rate)*100).toFixed(1)}</td><td>${r.profit_factor}</td><td>${r.avg_r}</td><td>${r.max_drawdown_pct}</td></tr>`;
  });
  h+="</tbody></table>";
  $("matrixResults").innerHTML=h;
}

async function poll(kind){
  try{
    const r=await fetch("/job/"+kind,{cache:"no-store"});
    const d=await r.json();
    setProgress(kind,d);
    if(d.metrics) renderMetrics(d.metrics);
    if(d.results) renderMatrix(d.results);
    if(d.data_exists) $("dataState").textContent="✅ Data ready";
    if(d.running) setTimeout(()=>poll(kind),800);
  }catch(e){
    $(kind+"Msg").textContent="状态读取失败，请稍后刷新";
    $(kind+"Btn").disabled=false;
  }
}

$("downloadBtn").addEventListener("click", async ()=>{
  const body=new URLSearchParams({start:$("start").value,end:$("end").value});
  const d=await (await fetch("/start-download",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body})).json();
  setProgress("download",d); poll("download");
});

$("backtestBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-backtest",{method:"POST"})).json();
  setProgress("backtest",d); poll("backtest");
});

$("matrixBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-matrix",{method:"POST"})).json();
  setProgress("matrix",d); poll("matrix");
});

poll("download"); poll("backtest"); poll("matrix");
</script>
</body>
</html>
"""

def default_dates():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365)
    return start.isoformat(), end.isoformat()

def set_job(kind, **kwargs):
    with LOCK:
        JOBS[kind].update(kwargs)

def render(**kwargs):
    start, end = default_dates()
    base = dict(start=start, end=end, preflight=None, data_exists=DATA.exists())
    base.update(kwargs)
    return render_template_string(HTML, **base)

@app.get("/")
def index():
    return render()

@app.post("/preflight")
def preflight():
    checks=[]; ok=True
    for f in ["engine.py","run_backtest.py","prepare_tiingo_data.py","config.json"]:
        exists=(BASE/f).exists()
        checks.append(f"{'OK' if exists else 'FAIL'} file {f}")
        ok = ok and exists
    token=bool(os.getenv("TIINGO_TOKEN"))
    checks.append(f"{'OK' if token else 'FAIL'} TIINGO_TOKEN")
    ok = ok and token
    try:
        import numpy, requests
        checks.append("OK python dependencies")
    except Exception as e:
        checks.append(f"FAIL dependencies: {e}")
        ok=False
    return render(preflight={"ok":ok,"text":"\n".join(checks)})

def download_worker(start,end):
    import subprocess
    try:
        set_job("download",running=True,progress=1,message="开始下载 20 只股票…",ok=None)
        p=subprocess.Popen(
            [sys.executable,"prepare_tiingo_data.py","--start",start,"--end",end,"--freq","5min","--out",str(DATA)],
            cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=os.environ.copy(),bufsize=1
        )
        lines=[]
        for line in p.stdout:
            line=line.rstrip(); lines.append(line)
            if line.startswith("[") and "/20]" in line:
                try:
                    n=int(line.split("/20]")[0][1:])
                    set_job("download",progress=max(2,int(n/20*90)),message=f"正在下载 {n}/20")
                except Exception:
                    pass
            elif line.startswith("PREFLIGHT"):
                set_job("download",progress=94,message="下载完成，正在做数据 Preflight…")
        rc=p.wait()
        if rc==0:
            set_job("download",running=False,progress=100,message="✅ 下载 + Preflight 完成",ok=True)
        else:
            set_job("download",running=False,progress=100,message="❌ 下载失败\n"+"\n".join(lines[-40:]),ok=False)
    except Exception as e:
        set_job("download",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

def backtest_worker():
    if not DATA.exists():
        set_job("backtest",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        set_job("backtest",running=True,progress=12,message="读取数据…",ok=None)
        df=pd.read_csv(DATA)
        cfg=json.loads((BASE/"config.json").read_text(encoding="utf-8"))
        set_job("backtest",progress=45,message="扫描 Gap + ORB + RVOL…")
        _,metrics=backtest(df,cfg)
        OUT.mkdir(exist_ok=True)
        (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        set_job("backtest",running=False,progress=100,message="✅ Baseline 完成",ok=True,metrics=metrics)
    except Exception as e:
        set_job("backtest",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

def matrix_worker():
    if not DATA.exists():
        set_job("matrix",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        set_job("matrix",running=True,progress=1,message="载入数据，只读一次…",ok=None,results=[])
        df=pd.read_csv(DATA)
        cfg=json.loads((BASE/"config.json").read_text(encoding="utf-8"))
        combos=[(r,g,o) for r in [1.2,1.5,2.0,3.0] for g in [.03,.04,.06] for o in [5,15,20]]
        rows=[]
        for i,(r,g,o) in enumerate(combos,1):
            c=dict(cfg)
            c["min_rvol"]=r
            c["min_gap_pct"]=g
            c["opening_range_minutes"]=o
            _,m=backtest(df,c)
            pf=m.get("profit_factor",0)
            if pf=="inf":
                pf=999.0
            row={
                "rvol":r,"gap":g,"or_min":o,
                "trades":m.get("trades",0),
                "win_rate":m.get("win_rate",0),
                "profit_factor":float(pf),
                "avg_r":m.get("avg_r",0),
                "max_drawdown_pct":m.get("max_drawdown_pct",0)
            }
            rows.append(row)
            ranked=sorted(rows,key=lambda x:(x["profit_factor"],x["trades"]),reverse=True)
            set_job("matrix",
                    progress=round(i/len(combos)*100),
                    message=f"正在跑 {i}/36：RVOL {r} / Gap {int(g*100)}% / OR {o}m",
                    results=ranked)
        rows=sorted(rows,key=lambda x:(x["profit_factor"],x["trades"]),reverse=True)
        OUT.mkdir(exist_ok=True)
        (OUT/"challenger_matrix.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        set_job("matrix",running=False,progress=100,
                message="✅ 36/36 完成。看稳定参数区域，不以单点最高 PF 下结论。",
                ok=True,results=rows)
    except Exception as e:
        set_job("matrix",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-download")
def start_download():
    with LOCK:
        if JOBS["download"]["running"]:
            return jsonify(JOBS["download"])
    start=request.form.get("start") or default_dates()[0]
    end=request.form.get("end") or default_dates()[1]
    threading.Thread(target=download_worker,args=(start,end),daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["download"])

@app.post("/start-backtest")
def start_backtest():
    with LOCK:
        if JOBS["backtest"]["running"]:
            return jsonify(JOBS["backtest"])
    threading.Thread(target=backtest_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["backtest"])

@app.post("/start-matrix")
def start_matrix():
    with LOCK:
        if JOBS["matrix"]["running"]:
            return jsonify(JOBS["matrix"])
    threading.Thread(target=matrix_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["matrix"])

@app.get("/job/<kind>")
def job(kind):
    if kind not in JOBS:
        return jsonify({"error":"unknown job"}),404
    with LOCK:
        d=dict(JOBS[kind])
    d["data_exists"]=DATA.exists()
    return jsonify(d)

@app.get("/health")
def health():
    return {"status":"ok"},200

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT","8080")))
