import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import date, timedelta

from flask import Flask, request, render_template_string

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "daytrade_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

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
input{padding:10px;border:1px solid #ccd2da;border-radius:8px;font-size:15px;width:150px}
pre{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;padding:12px;border-radius:10px;font-size:13px}
.metric{display:inline-block;min-width:120px;margin:6px;padding:10px;background:#f7f8fa;border-radius:10px}
</style>
</head>
<body><div class="wrap">
<h1>🇺🇸 US DayTrade Radar</h1>

<div class="card">
<h2>1. Preflight</h2>
<p class="muted">5秒内先检查文件、依赖、Tiingo Token 和接口，失败就不允许长跑。</p>
<form method="post" action="/preflight"><button>Run Preflight</button></form>
{% if preflight %}<pre class="{{ 'ok' if preflight.ok else 'bad' }}">{{ preflight.text }}</pre>{% endif %}
</div>

<div class="card">
<h2>2. 下载 Tiingo 样本</h2>
<form method="post" action="/download">
<label>Start <input type="date" name="start" value="{{ start }}"></label>
<label>End <input type="date" name="end" value="{{ end }}"></label><br>
<button>Download 20 Stocks / 5min</button>
</form>
{% if download_text %}<pre>{{ download_text }}</pre>{% endif %}
</div>

<div class="card">
<h2>3. ORB Backtest</h2>
<form method="post" action="/backtest"><button>Run Backtest</button></form>
{% if metrics %}
<div>{% for k,v in metrics.items() %}<div class="metric"><b>{{k}}</b><br>{{v}}</div>{% endfor %}</div>
{% endif %}
{% if backtest_text %}<pre>{{ backtest_text }}</pre>{% endif %}
</div>

<div class="card">
<h3>状态</h3>
<p>Data: {{ "✅ daytrade_data.csv" if data_exists else "❌ 未下载" }}</p>
<p>Results: {{ "✅ outputs/metrics.json" if metrics_exists else "— 尚未回测" }}</p>
<p class="muted">研究 / Paper Trading 阶段。不会发送 IBKR 真实订单。</p>
</div>
</div></body></html>
"""

def default_dates():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=35)
    return start.isoformat(), end.isoformat()

def run_cmd(args, timeout=180):
    p = subprocess.run(args, cwd=BASE_DIR, capture_output=True, text=True,
                       timeout=timeout, env=os.environ.copy())
    text = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, text[-12000:]

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
    base = dict(start=start, end=end, preflight=None, download_text=None,
                backtest_text=None, metrics=load_metrics(),
                data_exists=DATA_FILE.exists(),
                metrics_exists=(OUTPUT_DIR/"metrics.json").exists())
    base.update(kwargs)
    return render_template_string(HTML, **base)

@app.get("/")
def index():
    return render()

@app.post("/preflight")
def preflight():
    checks = []
    ok = True
    for f in ["engine.py", "run_backtest.py", "prepare_tiingo_data.py", "config.json"]:
        exists = (BASE_DIR / f).exists()
        checks.append(f"{'OK' if exists else 'FAIL'} file {f}")
        ok = ok and exists

    token = bool(os.getenv("TIINGO_TOKEN"))
    checks.append(f"{'OK' if token else 'FAIL'} TIINGO_TOKEN")
    ok = ok and token

    try:
        import pandas, numpy, requests
        checks.append("OK python dependencies")
    except Exception as e:
        checks.append(f"FAIL dependencies: {e}")
        ok = False

    if ok:
        start, end = default_dates()
        rc, txt = run_cmd([
            sys.executable, "prepare_tiingo_data.py",
            "--start", start, "--end", end,
            "--freq", "5min",
            "--symbols", "AAPL",
            "--out", "/tmp/daytrade_preflight.csv"
        ], timeout=30)
        if rc == 0:
            checks.append("OK Tiingo live sample + schema")
        else:
            ok = False
            checks.append("FAIL Tiingo live sample/schema")
            checks.append(txt[-2500:])

    return render(preflight={"ok": ok, "text": "\n".join(checks)})

@app.post("/download")
def download():
    start = request.form.get("start") or default_dates()[0]
    end = request.form.get("end") or default_dates()[1]
    rc, txt = run_cmd([
        sys.executable, "prepare_tiingo_data.py",
        "--start", start, "--end", end,
        "--freq", "5min",
        "--out", str(DATA_FILE)
    ], timeout=240)
    return render(start=start, end=end,
                  download_text=("SUCCESS\n" if rc == 0 else "FAILED\n") + txt)

@app.post("/backtest")
def backtest():
    if not DATA_FILE.exists():
        return render(backtest_text="FAILED: 先下载 daytrade_data.csv")
    rc, txt = run_cmd([
        sys.executable, "run_backtest.py",
        "--data", str(DATA_FILE)
    ], timeout=240)
    return render(backtest_text=("SUCCESS\n" if rc == 0 else "FAILED\n") + txt,
                  metrics=load_metrics())

@app.get("/health")
def health():
    return {"status":"ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
