import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import date, timedelta

from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "daytrade_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

JOBS = {
    "download": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "backtest": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
}
JOB_LOCK = threading.Lock()

HTML = """
<!doctype html>
<html lang="zh">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>US DayTrade Radar</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;background:#f4f6f8;margin:0;padding:18px;color:#111}
.wrap{max-width:860px;margin:auto}.card{background:#fff;border-radius:16px;padding:18px;margin:14px 0;box-shadow:0 1px 7px rgba(0,0,0,.08)}
h1{font-size:28px;margin:4px 0 18px}.ok{color:#0a8f42}.bad{color:#c62828}.muted{color:#667085}
button{background:#111827;color:#fff;border:0;border-radius:10px;padding:12px 16px;font-size:16px;margin:6px 6px 6px 0}
button:disabled{opacity:.45}
input{padding:10px;border:1px solid #ccd2da;border-radius:8px;font-size:16px;width:150px}
pre{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;padding:12px;border-radius:10px;font-size:13px}
.metric{display:inline-block;min-width:120px;margin:6px;padding:10px;background:#f7f8fa;border-radius:10px}
.progress-wrap{margin-top:14px}
.progress-row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:7px}
.progress-track{height:16px;border-radius:999px;background:#e5e7eb;overflow:hidden}
.progress-bar{height:100%;width:0%;background:#111827;transition:width .35s ease}
.progress-bar.done{background:#0a8f42}
.progress-bar.fail{background:#c62828}
.progress-msg{font-size:13px;color:#667085;margin-top:7px;word-break:break-word}
</style>
</head>
<body><div class="wrap">
<h1>🇺🇸 US DayTrade Radar</h1>

<div class="card">
<h2>1. Preflight</h2>
<p class="muted">先检查文件、依赖、Tiingo Token 和接口；失败就不允许长跑。</p>
<form method="post" action="/preflight"><button>Run Preflight</button></form>
{% if preflight %}<pre class="{{ 'ok' if preflight.ok else 'bad' }}">{{ preflight.text }}</pre>{% endif %}
</div>

<div class="card">
<h2>2. 下载 Tiingo 样本</h2>
<label>Start <input id="start" type="date" value="{{ start }}"></label>
<label>End <input id="end" type="date" value="{{ end }}"></label><br>
<button id="downloadBtn" type="button">Download 20 Stocks / 5min</button>

<div class="progress-wrap">
  <div class="progress-row"><b>下载进度</b><span id="downloadPct">0%</span></div>
  <div class="progress-track"><div id="downloadBar" class="progress-bar"></div></div>
  <div id="downloadMsg" class="progress-msg">尚未开始</div>
</div>
</div>

<div class="card">
<h2>3. ORB Backtest</h2>
<button id="backtestBtn" type="button">Run Backtest</button>

<div class="progress-wrap">
  <div class="progress-row"><b>回测进度</b><span id="backtestPct">0%</span></div>
  <div class="progress-track"><div id="backtestBar" class="progress-bar"></div></div>
  <div id="backtestMsg" class="progress-msg">尚未开始</div>
</div>

<div id="metricsBox" style="margin-top:12px">
{% if metrics %}
<div>{% for k,v in metrics.items() %}<div class="metric"><b>{{k}}</b><br>{{v}}</div>{% endfor %}</div>
{% endif %}
</div>
</div>

<div class="card">
<h3>状态</h3>
<p>Data: <span id="dataState">{{ "✅ daytrade_data.csv" if data_exists else "❌ 未下载" }}</span></p>
<p>Results: <span id="resultState">{{ "✅ outputs/metrics.json" if metrics_exists else "— 尚未回测" }}</span></p>
<p class="muted">研究 / Paper Trading 阶段。不会发送 IBKR 真实订单。</p>
</div>
</div>

<script>
const $ = (id)=>document.getElementById(id);
let timers = {};

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
  const holder=document.createElement("div");
  Object.entries(metrics).forEach(([k,v])=>{
    const d=document.createElement("div");
    d.className="metric";
    const b=document.createElement("b");
    b.textContent=k;
    d.appendChild(b);
    d.appendChild(document.createElement("br"));
    d.appendChild(document.createTextNode(String(v)));
    holder.appendChild(d);
  });
  box.appendChild(holder);
}

async function poll(kind){
  try{
    const r=await fetch("/job-status/"+kind,{cache:"no-store"});
    const d=await r.json();
    setProgress(kind,d);
    if(d.metrics) renderMetrics(d.metrics);
    if(d.data_exists) $("dataState").textContent="✅ daytrade_data.csv";
    if(d.metrics_exists) $("resultState").textContent="✅ outputs/metrics.json";
    if(d.running){
      timers[kind]=setTimeout(()=>poll(kind),800);
    }
  }catch(e){
    $(kind+"Msg").textContent="状态读取失败，请稍后刷新";
    $(kind+"Btn").disabled=false;
  }
}

$("downloadBtn").addEventListener("click", async ()=>{
  const body=new URLSearchParams({start:$("start").value,end:$("end").value});
  const r=await fetch("/start-download",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body});
  const d=await r.json();
  setProgress("download",d);
  poll("download");
});

$("backtestBtn").addEventListener("click", async ()=>{
  const r=await fetch("/start-backtest",{method:"POST"});
  const d=await r.json();
  setProgress("backtest",d);
  poll("backtest");
});

poll("download");
poll("backtest");
</script>
</body></html>
"""

def default_dates():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=35)
    return start.isoformat(), end.isoformat()

def run_cmd_stream(args, on_line=None, timeout=600):
    p = subprocess.Popen(args, cwd=BASE_DIR, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         env=os.environ.copy(), bufsize=1)
    lines=[]
    started=time.time()
    for line in p.stdout:
        lines.append(line.rstrip())
        if on_line:
            on_line(line.rstrip())
        if time.time()-started > timeout:
            p.kill()
            raise TimeoutError("job timeout")
    rc=p.wait()
    return rc, "\n".join(lines[-300:])

def load_metrics():
    p = OUTPUT_DIR / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def render(**kwargs):
    start, end = default_dates()
    base = dict(start=start,end=end,preflight=None,metrics=load_metrics(),
                data_exists=DATA_FILE.exists(),
                metrics_exists=(OUTPUT_DIR/"metrics.json").exists())
    base.update(kwargs)
    return render_template_string(HTML, **base)

def set_job(kind, **kwargs):
    with JOB_LOCK:
        JOBS[kind].update(kwargs)

@app.get("/")
def index():
    return render()

@app.post("/preflight")
def preflight():
    checks=[]; ok=True
    for f in ["engine.py","run_backtest.py","prepare_tiingo_data.py","config.json"]:
        exists=(BASE_DIR/f).exists()
        checks.append(f"{'OK' if exists else 'FAIL'} file {f}")
        ok = ok and exists
    token=bool(os.getenv("TIINGO_TOKEN"))
    checks.append(f"{'OK' if token else 'FAIL'} TIINGO_TOKEN")
    ok = ok and token
    try:
        import pandas, numpy, requests
        checks.append("OK python dependencies")
    except Exception as e:
        checks.append(f"FAIL dependencies: {e}"); ok=False
    if ok:
        start,end=default_dates()
        p=subprocess.run([sys.executable,"prepare_tiingo_data.py","--start",start,"--end",end,
                          "--freq","5min","--symbols","AAPL","--out","/tmp/daytrade_preflight.csv"],
                         cwd=BASE_DIR,capture_output=True,text=True,timeout=40,env=os.environ.copy())
        if p.returncode==0:
            checks.append("OK Tiingo live sample + schema")
        else:
            ok=False
            checks.append("FAIL Tiingo live sample/schema")
            checks.append(((p.stdout or "")+(p.stderr or ""))[-1800:])
    return render(preflight={"ok":ok,"text":"\n".join(checks)})

def download_worker(start,end):
    try:
        set_job("download",running=True,progress=1,message="开始下载 20 只股票…",ok=None)
        seen=set()
        def handle(line):
            if line.startswith("[") and "/20]" in line:
                try:
                    n=int(line.split("/20]")[0].lstrip("["))
                    seen.add(n)
                    set_job("download",progress=max(2,int(n/20*90)),
                            message=f"正在下载 {n}/20：{line}")
                except Exception:
                    pass
            elif line.startswith("PREFLIGHT"):
                set_job("download",progress=93,message="下载完成，正在做数据 Preflight…")
            elif line.startswith("rows:"):
                set_job("download",progress=96,message=f"Preflight：{line}")
            elif line.startswith("Saved:"):
                set_job("download",progress=99,message=line)

        rc,text=run_cmd_stream([sys.executable,"prepare_tiingo_data.py","--start",start,"--end",end,
                                "--freq","5min","--out",str(DATA_FILE)],handle,timeout=600)
        if rc==0:
            set_job("download",running=False,progress=100,message="✅ 下载与 Preflight 全部完成",ok=True)
        else:
            set_job("download",running=False,progress=100,message="❌ 下载失败\n"+text[-2500:],ok=False)
    except Exception as e:
        set_job("download",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

def backtest_worker():
    try:
        if not DATA_FILE.exists():
            set_job("backtest",running=False,progress=100,message="❌ 先下载 daytrade_data.csv",ok=False)
            return
        set_job("backtest",running=True,progress=8,message="读取真实分钟数据…",ok=None)
        time.sleep(.25)
        set_job("backtest",progress=22,message="构建 RVOL 历史基线…")
        time.sleep(.25)
        set_job("backtest",progress=38,message="扫描 Gap + Opening Range…")
        time.sleep(.25)
        set_job("backtest",progress=55,message="生成 ORB + RVOL 信号…")
        time.sleep(.25)
        set_job("backtest",progress=72,message="模拟 Entry / Stop / Target / EOD…")
        rc,text=run_cmd_stream([sys.executable,"run_backtest.py","--data",str(DATA_FILE)],timeout=600)
        if rc==0:
            set_job("backtest",running=False,progress=100,message="✅ 回测完成",ok=True)
        else:
            set_job("backtest",running=False,progress=100,message="❌ 回测失败\n"+text[-2500:],ok=False)
    except Exception as e:
        set_job("backtest",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-download")
def start_download():
    with JOB_LOCK:
        if JOBS["download"]["running"]:
            return jsonify(JOBS["download"])
    start=request.form.get("start") or default_dates()[0]
    end=request.form.get("end") or default_dates()[1]
    threading.Thread(target=download_worker,args=(start,end),daemon=True).start()
    time.sleep(.05)
    with JOB_LOCK:
        return jsonify(JOBS["download"])

@app.post("/start-backtest")
def start_backtest():
    with JOB_LOCK:
        if JOBS["backtest"]["running"]:
            return jsonify(JOBS["backtest"])
    threading.Thread(target=backtest_worker,daemon=True).start()
    time.sleep(.05)
    with JOB_LOCK:
        return jsonify(JOBS["backtest"])

@app.get("/job-status/<kind>")
def job_status(kind):
    if kind not in JOBS:
        return jsonify({"error":"unknown job"}),404
    with JOB_LOCK:
        d=dict(JOBS[kind])
    d["data_exists"]=DATA_FILE.exists()
    d["metrics_exists"]=(OUTPUT_DIR/"metrics.json").exists()
    if kind=="backtest":
        d["metrics"]=load_metrics()
    return jsonify(d)

@app.get("/health")
def health():
    return {"status":"ok"},200

if __name__=="__main__":
    port=int(os.environ.get("PORT","8080"))
    app.run(host="0.0.0.0",port=port)
