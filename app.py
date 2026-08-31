import json, os, sys, threading, time, re
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from engine import backtest
from filter_ladder import run_filter_ladder
from winner_loser_diag import run_winner_loser_diagnostic
from entry_quality_v06 import run_entry_quality_validation
from robustness_v07 import run_robustness_v07
from true_oos_v08 import run_true_oos_v08

app = Flask(__name__)

BASE = Path(__file__).resolve().parent

# Persistent Railway volume.
# Mount a Railway Volume at /data, then market data/cache/results survive deploys.
DATA_ROOT = Path(os.getenv("DAYTRADE_DATA_DIR", "/data/daytrade"))
try:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_ROOT = BASE / "runtime_data"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

DATA = DATA_ROOT / "daytrade_data.csv"
SIP_DATA = DATA_ROOT / "sip_data.csv"
SIP_CACHE = DATA_ROOT / "sip_cache"
OUT = DATA_ROOT / "outputs"
OOS_DIR = DATA_ROOT / "oos_v08"
OOS_DATA = OOS_DIR / "sip_oos_2024-08-30_2025-08-29.csv"
OOS_CACHE = OOS_DIR / "sip_cache"
OOS_OUT = OOS_DIR / "outputs"
OOS_DIR.mkdir(parents=True, exist_ok=True)
OOS_CACHE.mkdir(parents=True, exist_ok=True)
OOS_OUT.mkdir(parents=True, exist_ok=True)
SIP_CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

LOCK = threading.Lock()

JOBS = {
    "download": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "backtest": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "matrix": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "results": []},
    "ladder": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "results": []},
    "dna": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "entryq": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "robust": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "oosdl": {"running": False, "progress": 0, "message": "尚未开始", "ok": None},
    "oos": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
}

HTML = """
<!doctype html>
<html lang="zh">
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>US DayTrade Radar v0.8.1</title>
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
<h1>🇺🇸 US DayTrade Radar <small>v0.8.1</small></h1>

<div class="card">
<h2>1. Preflight</h2>
<p class="muted">先检查文件、依赖、Tiingo Token 和接口；失败就不允许长跑。</p>
<form method="post" action="/preflight"><button>Run Preflight</button></form>
{% if preflight %}
<pre class="{{ 'ok' if preflight.ok else 'bad' }}">{{ preflight.text }}</pre>
{% endif %}
</div>

<div class="card">
<h2>2. Stocks-in-Play 125</h2>
<label>Start <input id="start" type="date" value="{{ start }}"></label>
<label>End <input id="end" type="date" value="{{ end }}"></label><br>
<button id="downloadBtn" type="button">Download 125 Stocks-in-Play / 5min</button>
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
<h2>5. V0.4 Filter Ladder</h2>
<p><b>冻结基准：</b> RVOL 3.0 / Gap ≥3% / OR 20min</p>
<p class="muted">只做预先定义的累加过滤，不扫阈值：Baseline → +VWAP → +Breakout Volume → +Close Strength → +OR Quality。每层显示 Trades Retention / Win% / PF / Avg R / MaxDD，直接看哪一刀真正增加 edge。</p>
<button id="ladderBtn" type="button">Run V0.4 Filter Ladder</button>
<div class="row"><b>Filter Ladder 进度</b><span id="ladderPct">0%</span></div>
<div class="track"><div id="ladderBar" class="bar"></div></div>
<div id="ladderMsg" class="msg">尚未开始</div>
<div id="ladderResults" class="scroll"></div>
</div>

<div class="card">
<h2>6. V0.5 Winner vs Loser DNA</h2>
<p><b>冻结研究基准：</b> RVOL 3.0 / Gap ≥3% / OR20 / +VWAP</p>
<p class="muted">不扫阈值。只比较 Winner vs Loser / Target vs Stop 的入场前特征，并把 MFE/MAE 只作为未来标签。输出 Gap、RVOL、VWAP距离、OR宽度、突破时间、突破幅度、前三根走势与量能差异。</p>
<button id="dnaBtn" type="button">Run Winner vs Loser DNA</button>
<div class="row"><b>DNA 进度</b><span id="dnaPct">0%</span></div>
<div class="track"><div id="dnaBar" class="bar"></div></div>
<div id="dnaMsg" class="msg">尚未开始</div>
<div id="dnaResults" class="scroll"></div>
</div>

<div class="card">
<h2>7. V0.6 Entry Quality Validation</h2>
<p><b>冻结基准：</b> RVOL 3.0 / Gap ≥3% / OR20 / +VWAP</p>
<p class="muted">不重新下载、不重跑 36 Matrix。直接读取 V0.5 DNA Trade Cache，只验证 3 个预先定义的 Gate：Anti-Chase（前三根涨幅 ≤0.75%）、OR Quality（OR宽度 ≥3.5%）、两者组合。不是阈值扫描。</p>
<button id="entryqBtn" type="button">Run V0.6 Entry Quality</button>
<div class="row"><b>Entry Quality 进度</b><span id="entryqPct">0%</span></div>
<div class="track"><div id="entryqBar" class="bar"></div></div>
<div id="entryqMsg" class="msg">尚未开始</div>
<div id="entryqResults" class="scroll"></div>
</div>

<div class="card">
<h2>8. V0.7 Robustness Validation</h2>
<p><b>冻结 Gate：</b> prior 3-bar return ≤ 0.75% AND OR range ≥ 3.50%</p>
<p class="muted">不调阈值、不重新下载、不重跑 Matrix。直接用 V0.5 DNA Trade Cache 做 4 个时间块 + 4 个固定 Symbol Buckets + 3000 次 Bootstrap。此阶段只检查稳定性，不把同一年数据冒充真正 OOS。</p>
<button id="robustBtn" type="button">Run V0.7 Robustness</button>
<div class="row"><b>Robustness 进度</b><span id="robustPct">0%</span></div>
<div class="track"><div id="robustBar" class="bar"></div></div>
<div id="robustMsg" class="msg">尚未开始</div>
<div id="robustResults" class="scroll"></div>
</div>

<div class="card">
<h2>9. V0.8 True OOS Validation</h2>
<p><b>锁死规则：</b> Baseline = RVOL 3.0 / Gap ≥3% / OR20 / +VWAP；Gate = prior 3-bar return ≤0.75% AND OR range ≥3.50%</p>
<p class="muted">Untouched OOS 固定为 2024-08-30 → 2025-08-29，与发现/robustness 年 2025-08-30 → 2026-08-30 完全分开。只下载这一次并写入 /data/daytrade/oos_v08，之后部署继续复用。看完 OOS 后禁止再调 0.75% / 3.50%。</p>
<button id="oosDlBtn" type="button">Download Untouched OOS 125 / 5min</button>
<div class="row"><b>OOS 下载进度</b><span id="oosdlPct">0%</span></div>
<div class="track"><div id="oosdlBar" class="bar"></div></div>
<div id="oosdlMsg" class="msg">尚未开始</div>

<button id="oosBtn" type="button">Run V0.8 True OOS</button>
<div class="row"><b>True OOS 进度</b><span id="oosPct">0%</span></div>
<div class="track"><div id="oosBar" class="bar"></div></div>
<div id="oosMsg" class="msg">尚未开始</div>
<div id="oosResults" class="scroll"></div>
</div>

<div class="card">
<h3>状态</h3>
<p>Data:
{% if persistent_data.replay_ready %}
<span id="dataState">✅ Persistent Replay ready</span>
{% elif sip_data_exists %}
<span id="dataState">✅ SIP 125 Data ready</span>
{% elif data_exists %}
<span id="dataState">✅ 20-stock Data ready</span>
{% else %}
<span id="dataState">❌ No data</span>
{% endif %}
</p>
<p>Storage: <b>{{ data_root }}</b></p>
<p>Replay file: <b>{{ persistent_data.sip_size_mb }} MB</b></p>
<p>Cache: <b>{{ persistent_data.cache_files }} files / {{ persistent_data.cache_size_mb }} MB</b></p>
<p class="muted">Deploy 后会自动扫描 /data；已有 Replay/Cache 时直接复用，不再要求重新下载。</p>
<button id="rescanBtn" type="button">Rescan Persistent Data</button>
<div id="rescanMsg" class="msg"></div>
<p class="muted">Research / Paper only. No IBKR live orders.</p>
</div>
</div>

<script>
const $ = (id)=>document.getElementById(id);

function setProgress(kind, data){
  const bar=$(kind+"Bar"), pct=$(kind+"Pct"), msg=$(kind+"Msg");
  // oosdl historically used oosDlBtn (capital D), so never assume kind+"Btn" exists.
  const btn=$(kind+"Btn") || (kind==="oosdl" ? $("oosDlBtn") : null);
  const p=Math.max(0,Math.min(100,Number(data.progress||0)));
  if(bar) bar.style.width=p+"%";
  if(pct) pct.textContent=p+"%";
  if(msg) msg.textContent=data.message||"";
  if(btn) btn.disabled=!!data.running;
  if(bar){
    bar.classList.remove("done","fail");
    if(!data.running && data.ok===true) bar.classList.add("done");
    if(!data.running && data.ok===false) bar.classList.add("fail");
  }
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


function renderLadder(rows){
  if(!rows || !rows.length) return;
  let h='<table><thead><tr><th>Step</th><th>Trades</th><th>Retain%</th><th>Win%</th><th>PF</th><th>Avg R</th><th>MaxDD%</th><th>Net PnL</th></tr></thead><tbody>';
  rows.forEach((r)=>{
    const pf=Number(r.profit_factor);
    const cls=(pf>=1.2 && Number(r.trades)>=80 && Number(r.avg_r)>0) ? "pass" : "";
    h+=`<tr class="${cls}"><td>${r.name}</td><td>${r.trades}</td><td>${r.retention_pct}</td><td>${(Number(r.win_rate)*100).toFixed(1)}</td><td>${r.profit_factor}</td><td>${r.avg_r}</td><td>${r.max_drawdown_pct}</td><td>${r.net_pnl}</td></tr>`;
  });
  h+="</tbody></table>";
  $("ladderResults").innerHTML=h;
}

function renderDNA(r){
  if(!r) return;
  let h=`<p><b>${r.frozen_rule}</b></p><p>Trades ${r.trades} · Win ${(Number(r.win_rate)*100).toFixed(1)}% · PF ${r.profit_factor} · Avg R ${r.avg_r} · MaxDD ${r.max_drawdown_pct}%</p>`;
  h+=`<p>Winner ${r.winner_count} / Loser ${r.loser_count} · Target ${r.target_count} / Stop ${r.stop_count}</p>`;
  h+='<table><thead><tr><th>Feature</th><th>Winner Med</th><th>Loser Med</th><th>Delta</th><th>Effect</th></tr></thead><tbody>';
  (r.features||[]).forEach(x=>{h+=`<tr><td>${x.feature}</td><td>${x.winner_median}</td><td>${x.loser_median}</td><td>${x.delta}</td><td>${x.robust_effect}</td></tr>`});
  h+='</tbody></table>';
  h+=`<p class="muted">MFE median: Winner ${r.winner_mfe_r_median}R / Loser ${r.loser_mfe_r_median}R. MAE median: Winner ${r.winner_mae_r_median}R / Loser ${r.loser_mae_r_median}R. MFE/MAE 只作标签，不可作为入场条件。</p>`;
  $("dnaResults").innerHTML=h;
}


function renderEntryQ(r){
  if(!r) return;
  let h=`<p><b>${r.study}</b></p><p>${r.frozen_baseline}</p>`;
  h+='<table><thead><tr><th>Gate</th><th>Trades</th><th>Retain%</th><th>Win%</th><th>PF</th><th>ΔPF</th><th>Avg R</th><th>Stop%</th><th>MaxDD%</th><th>H1 PF</th><th>H2 PF</th><th>Verdict</th></tr></thead><tbody>';
  (r.rows||[]).forEach(x=>{
    const cls=(x.verdict==="PASS")?"pass":"";
    h+=`<tr class="${cls}"><td>${x.name}<br><span class="muted">${x.rule}</span></td><td>${x.trades}</td><td>${x.retention_pct}</td><td>${(Number(x.win_rate)*100).toFixed(1)}</td><td>${x.profit_factor}</td><td>${x.delta_pf}</td><td>${x.avg_r}</td><td>${(Number(x.stop_rate)*100).toFixed(1)}</td><td>${x.max_drawdown_pct}</td><td>${x.h1_pf}</td><td>${x.h2_pf}</td><td><b>${x.verdict}</b></td></tr>`;
  });
  h+='</tbody></table>';
  if(r.best_candidate){
    h+=`<p><b>Best validated candidate:</b> ${r.best_candidate.name} · ${r.best_candidate.verdict}</p>`;
  }else{
    h+='<p><b>结论：</b>没有 Gate 达到 PASS/WATCH；保留 V0.5 Frozen Baseline，不强行优化。</p>';
  }
  h+=`<p class="muted">${r.decision_rule}</p>`;
  $("entryqResults").innerHTML=h;
}


function renderRobust(r){
  if(!r) return;
  let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total} checks</p>`;
  h+=`<p><b>Frozen Gate:</b> ${r.gate}</p>`;
  const o=r.overall||{};
  if(o.gate){
    h+=`<p><b>Overall:</b> ${o.gate.trades} trades · PF ${o.gate.profit_factor} · AvgR ${o.gate.avg_r} · Win ${(Number(o.gate.win_rate)*100).toFixed(1)}% · Stop ${(Number(o.gate.stop_rate)*100).toFixed(1)}% · Retain ${o.retention_pct}%</p>`;
  }
  const b=r.bootstrap||{};
  h+=`<p><b>Bootstrap:</b> AvgR 95% CI [${(b.avg_r_ci95||[]).join(", ")}] · PF 95% CI [${(b.pf_ci95||[]).join(", ")}] · P(AvgR>0) ${(Number(b.prob_avg_r_gt0||0)*100).toFixed(1)}% · P(PF>1) ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}%</p>`;

  h+='<h3>Time Blocks</h3><table><thead><tr><th>Block</th><th>Gate Trades</th><th>Retain%</th><th>Win%</th><th>PF</th><th>AvgR</th><th>Stop%</th></tr></thead><tbody>';
  (r.time_blocks||[]).forEach(x=>{
    h+=`<tr><td>${x.block}<br><span class="muted">${x.start}..${x.end}</span></td><td>${x.gate.trades}</td><td>${x.retention_pct}</td><td>${(Number(x.gate.win_rate)*100).toFixed(1)}</td><td>${x.gate.profit_factor}</td><td>${x.gate.avg_r}</td><td>${(Number(x.gate.stop_rate)*100).toFixed(1)}</td></tr>`;
  });
  h+='</tbody></table>';

  h+='<h3>Symbol Buckets</h3><table><thead><tr><th>Bucket</th><th>Symbols</th><th>Gate Trades</th><th>Retain%</th><th>Win%</th><th>PF</th><th>AvgR</th><th>Stop%</th></tr></thead><tbody>';
  (r.symbol_buckets||[]).forEach(x=>{
    h+=`<tr><td>${x.bucket}</td><td>${x.symbols}</td><td>${x.gate.trades}</td><td>${x.retention_pct}</td><td>${(Number(x.gate.win_rate)*100).toFixed(1)}</td><td>${x.gate.profit_factor}</td><td>${x.gate.avg_r}</td><td>${(Number(x.gate.stop_rate)*100).toFixed(1)}</td></tr>`;
  });
  h+='</tbody></table>';

  h+='<h3>Checks</h3><table><tbody>';
  Object.entries(r.checks||{}).forEach(([k,v])=>{
    h+=`<tr><td>${k}</td><td><b>${v ? "✅ PASS" : "❌ FAIL"}</b></td></tr>`;
  });
  h+='</tbody></table>';
  h+=`<p class="muted">${r.same_sample_warning}</p><p><b>Next:</b> ${r.next_step}</p>`;
  $("robustResults").innerHTML=h;
}


function renderOOS(r){
  if(!r) return;
  let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total} checks</p>`;
  h+=`<p><b>Frozen Baseline:</b> ${r.frozen_baseline}<br><b>Frozen Gate:</b> ${r.frozen_gate}</p>`;
  h+='<table><thead><tr><th>Arm</th><th>Trades</th><th>Win%</th><th>PF</th><th>Avg R</th><th>Stop%</th><th>Target%</th><th>MaxDD%</th><th>Net PnL</th></tr></thead><tbody>';
  const b=r.baseline||{}, g=r.gate||{};
  h+=`<tr><td>Baseline</td><td>${b.trades}</td><td>${(Number(b.win_rate||0)*100).toFixed(1)}</td><td>${b.profit_factor}</td><td>${b.avg_r}</td><td>${(Number(b.stop_rate||0)*100).toFixed(1)}</td><td>${(Number(b.target_rate||0)*100).toFixed(1)}</td><td>${b.max_drawdown_pct}</td><td>${b.net_pnl}</td></tr>`;
  h+=`<tr><td>Frozen Gate</td><td>${g.trades}</td><td>${(Number(g.win_rate||0)*100).toFixed(1)}</td><td>${g.profit_factor}</td><td>${g.avg_r}</td><td>${(Number(g.stop_rate||0)*100).toFixed(1)}</td><td>${(Number(g.target_rate||0)*100).toFixed(1)}</td><td>${g.max_drawdown_pct}</td><td>${g.net_pnl}</td></tr>`;
  h+='</tbody></table>';
  h+=`<p><b>Retention:</b> ${r.retention_pct}% · <b>ΔPF:</b> ${r.delta_pf} · <b>ΔAvgR:</b> ${r.delta_avg_r} · <b>ΔStop:</b> ${r.delta_stop_rate_pp}pp</p>`;
  h+='<h3>Checks</h3><table><tbody>';
  Object.entries(r.checks||{}).forEach(([k,v])=>{
    h+=`<tr><td>${k}</td><td><b>${v ? "✅ PASS" : "❌ FAIL"}</b></td></tr>`;
  });
  h+='</tbody></table>';
  h+=`<p class="muted">${r.note}</p>`;
  $("oosResults").innerHTML=h;
}

async function poll(kind){
  try{
    const r=await fetch("/job/"+kind,{cache:"no-store"});
    const d=await r.json();
    setProgress(kind,d);
    if(d.metrics) renderMetrics(d.metrics);
    if(d.results){ if(kind==="ladder") renderLadder(d.results); else renderMatrix(d.results); }
    if(d.result && kind==="dna") renderDNA(d.result);
    if(d.result && kind==="entryq") renderEntryQ(d.result);
    if(d.result && kind==="robust") renderRobust(d.result);
    if(d.result && kind==="oos") renderOOS(d.result);
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


$("ladderBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-ladder",{method:"POST"})).json();
  setProgress("ladder",d); poll("ladder");
});

$("dnaBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-dna",{method:"POST"})).json();
  setProgress("dna",d); poll("dna");
});


$("entryqBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-entryq",{method:"POST"})).json();
  setProgress("entryq",d); poll("entryq");
});


$("robustBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-robust",{method:"POST"})).json();
  setProgress("robust",d); poll("robust");
});


$("oosDlBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-oos-download",{method:"POST"})).json();
  setProgress("oosdl",d); poll("oosdl");
});

$("oosBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-oos",{method:"POST"})).json();
  setProgress("oos",d); poll("oos");
});

$("rescanBtn").addEventListener("click", async ()=>{
  try{
    const d=await (await fetch("/rescan-data",{method:"POST"})).json();
    $("rescanMsg").textContent = `Replay ready: ${d.replay_ready ? "YES" : "NO"} | sip_data: ${d.sip_size_mb} MB | cache: ${d.cache_files} files / ${d.cache_size_mb} MB`;
    if(d.replay_ready) $("dataState").textContent="✅ Persistent Replay ready";
  }catch(e){
    $("rescanMsg").textContent="Rescan failed";
  }
});

poll("download"); poll("backtest"); poll("matrix"); poll("ladder"); poll("dna"); poll("entryq"); poll("robust"); poll("oosdl"); poll("oos");
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


def discover_persistent_data():
    info = {
        "data_root": str(DATA_ROOT),
        "volume_ok": False,
        "sip_exists": SIP_DATA.exists(),
        "sip_size_mb": 0.0,
        "cache_files": 0,
        "cache_size_mb": 0.0,
        "replay_ready": False,
    }

    try:
        probe = DATA_ROOT / ".volume_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        info["volume_ok"] = True
    except Exception:
        info["volume_ok"] = False

    if SIP_DATA.exists():
        try:
            info["sip_size_mb"] = round(SIP_DATA.stat().st_size / (1024 * 1024), 1)
        except Exception:
            pass

    try:
        files = [p for p in SIP_CACHE.glob("*.csv") if p.is_file()]
        info["cache_files"] = len(files)
        info["cache_size_mb"] = round(sum(p.stat().st_size for p in files) / (1024 * 1024), 1)
    except Exception:
        pass

    info["replay_ready"] = bool(info["sip_exists"] or info["cache_files"] > 0)
    return info



def discover_oos_data(deep=False):
    info = {
        "exists": OOS_DATA.exists(),
        "size_mb": 0.0,
        "cache_files": 0,
        "cache_size_mb": 0.0,
        "symbols": None,
        "rows": None,
        "duplicates": None,
        "bad_timestamp": None,
        "start": None,
        "end": None,
        "schema_ok": None,
        "ready": False,
    }

    if OOS_DATA.exists():
        try:
            info["size_mb"] = round(OOS_DATA.stat().st_size / (1024 * 1024), 1)
        except Exception:
            pass

    try:
        files = [p for p in OOS_CACHE.glob("*.csv") if p.is_file()]
        info["cache_files"] = len(files)
        info["cache_size_mb"] = round(sum(p.stat().st_size for p in files) / (1024 * 1024), 1)
    except Exception:
        pass

    # Fast readiness requires both merged file and the expected 125 symbol cache files.
    info["ready"] = bool(info["exists"] and info["size_mb"] > 1.0 and info["cache_files"] == 125)

    if deep and info["exists"]:
        try:
            probe = pd.read_csv(OOS_DATA)
            needed = {"symbol","timestamp","open","high","low","close","volume"}
            info["schema_ok"] = needed.issubset(set(probe.columns))
            info["rows"] = int(len(probe))
            if "symbol" in probe.columns:
                info["symbols"] = int(probe["symbol"].astype(str).nunique())
            if "timestamp" in probe.columns:
                ts = pd.to_datetime(probe["timestamp"], errors="coerce", utc=True)
                info["bad_timestamp"] = int(ts.isna().sum())
                if ts.notna().any():
                    info["start"] = str(ts.min())
                    info["end"] = str(ts.max())
            if {"symbol","timestamp"}.issubset(probe.columns):
                info["duplicates"] = int(probe.duplicated(["symbol","timestamp"]).sum())

            info["ready"] = bool(
                info["schema_ok"] and
                (info["symbols"] == 125) and
                (info["rows"] or 0) > 100000 and
                (info["bad_timestamp"] == 0) and
                (info["duplicates"] == 0)
            )
        except Exception:
            info["ready"] = False

    return info


def persistent_preflight_lines():
    d = discover_persistent_data()
    lines = []

    if d["volume_ok"]:
        lines.append(f"OK volume writable: {d['data_root']}")
    else:
        lines.append(f"FAIL volume not writable: {d['data_root']}")

    if d["sip_exists"]:
        lines.append(f"OK existing Replay Dataset: sip_data.csv ({d['sip_size_mb']} MB)")
    else:
        lines.append("INFO no merged sip_data.csv yet")

    if d["cache_files"] > 0:
        lines.append(f"OK existing SIP cache: {d['cache_files']} files / {d['cache_size_mb']} MB")
    else:
        lines.append("INFO no SIP cache files yet")

    lines.append(f"Reuse existing data: {'YES' if d['replay_ready'] else 'NO'}")
    return lines, d

def render(**kwargs):
    start, end = default_dates()
    pdata = discover_persistent_data()
    base = dict(
        start=start,
        end=end,
        preflight=None,
        data_exists=DATA.exists(),
        sip_data_exists=SIP_DATA.exists(),
        data_root=str(DATA_ROOT),
        persistent_data=pdata,
    )
    base.update(kwargs)
    return render_template_string(HTML, **base)

@app.get("/")
def index():
    return render()

@app.post("/preflight")
def preflight():
    checks=[]; ok=True
    for f in ["engine.py","run_backtest.py","prepare_tiingo_data.py","prepare_sip_data.py","filter_ladder.py","winner_loser_diag.py","entry_quality_v06.py","robustness_v07.py","true_oos_v08.py","config.json"]:
        exists=(BASE/f).exists()
        checks.append(f"{'OK' if exists else 'FAIL'} file {f}")
        ok = ok and exists
    token=bool(os.getenv("TIINGO_TOKEN"))
    checks.append(f"{'OK' if token else 'FAIL'} TIINGO_TOKEN")
    ok = ok and token

    try:
        probe = DATA_ROOT / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(f"OK storage writable: {DATA_ROOT}")
    except Exception as e:
        checks.append(f"FAIL storage not writable: {DATA_ROOT} ({e})")
        ok = False

    if str(DATA_ROOT).startswith("/app"):
        checks.append("WARN storage is under /app and will be erased on deploy")
    elif str(DATA_ROOT).startswith("/data"):
        checks.append("OK storage path is /data (persistent if Railway Volume is mounted)")
    try:
        import numpy, requests
        checks.append("OK python dependencies")
    except Exception as e:
        checks.append(f"FAIL dependencies: {e}")
        ok=False

    storage_lines, storage_info = persistent_preflight_lines()
    checks.extend(storage_lines)
    ok = ok and storage_info["volume_ok"]

    oos_info = discover_oos_data(deep=True)
    if oos_info["ready"]:
        checks.append(
            f"OK untouched OOS dataset: {OOS_DATA.name} "
            f"({oos_info['size_mb']} MB / {oos_info['rows']} rows / {oos_info['symbols']} symbols)"
        )
        checks.append(
            f"OK untouched OOS schema/timestamps/duplicates: "
            f"bad_timestamp={oos_info['bad_timestamp']} duplicates={oos_info['duplicates']}"
        )
        checks.append(
            f"OK untouched OOS cache: {oos_info['cache_files']} files / {oos_info['cache_size_mb']} MB"
        )
        checks.append("Reuse untouched OOS data: YES")
    elif oos_info["exists"] or oos_info["cache_files"] > 0:
        checks.append(
            f"FAIL untouched OOS incomplete: merged={oos_info['exists']} "
            f"size={oos_info['size_mb']}MB cache_files={oos_info['cache_files']} "
            f"rows={oos_info['rows']} symbols={oos_info['symbols']}"
        )
        checks.append("Reuse untouched OOS data: NO")
        ok = False
    else:
        checks.append("INFO untouched OOS dataset not downloaded yet")
        checks.append("Reuse untouched OOS data: NO")

    return render(preflight={"ok":ok,"text":"\n".join(checks)})

def download_worker(start,end):
    import subprocess
    try:
        set_job("download",running=True,progress=1,message="开始下载 125 只 Stocks-in-Play…",ok=None)
        p=subprocess.Popen(
            [sys.executable,"prepare_sip_data.py","--start",start,"--end",end,"--freq","5min","--out",str(SIP_DATA),"--cache-dir",str(SIP_CACHE)],
            cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=os.environ.copy(),bufsize=1
        )
        lines=[]
        for line in p.stdout:
            line=line.rstrip(); lines.append(line)
            if line.startswith("[") and "/125]" in line:
                try:
                    n=int(line.split("/125]")[0][1:])
                    set_job("download",progress=max(2,int(n/125*90)),message=f"正在下载 {n}/20")
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
    data_path = SIP_DATA if SIP_DATA.exists() else DATA
    if not data_path.exists():
        set_job("backtest",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        set_job("backtest",running=True,progress=12,message=f"读取数据：{data_path.name}…",ok=None)
        df=pd.read_csv(data_path)
        cfg=json.loads((BASE/"config.json").read_text(encoding="utf-8"))
        set_job("backtest",progress=45,message="扫描 Gap + ORB + RVOL…")
        _,metrics=backtest(df,cfg)
        OUT.mkdir(exist_ok=True)
        (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        set_job("backtest",running=False,progress=100,message="✅ Baseline 完成",ok=True,metrics=metrics)
    except Exception as e:
        set_job("backtest",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

def matrix_worker():
    data_path = SIP_DATA if SIP_DATA.exists() else DATA
    if not data_path.exists():
        set_job("matrix",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        set_job("matrix",running=True,progress=1,message=f"载入 {data_path.name}，只读一次…",ok=None,results=[])
        df=pd.read_csv(data_path)
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


def ladder_worker():
    data_path = SIP_DATA if SIP_DATA.exists() else DATA
    if not data_path.exists():
        set_job("ladder",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        set_job("ladder",running=True,progress=5,message="读取固定 Replay Dataset…",ok=None,results=[])
        results = run_filter_ladder(str(data_path), str(BASE/"config.json"))
        # Ladder is only 5 cumulative layers; report completion after deterministic run.
        OUT.mkdir(exist_ok=True)
        (OUT/"filter_ladder_v04.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        set_job("ladder",running=False,progress=100,
                message="✅ V0.4 Filter Ladder 完成。只看逐层增益，不做阈值拟合。",
                ok=True,results=results)
    except Exception as e:
        set_job("ladder",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-ladder")
def start_ladder():
    with LOCK:
        if JOBS["ladder"]["running"]:
            return jsonify(JOBS["ladder"])
    threading.Thread(target=ladder_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["ladder"])

def dna_worker():
    data_path = SIP_DATA if SIP_DATA.exists() else DATA
    if not data_path.exists():
        set_job("dna",running=False,progress=100,message="❌ 先下载数据",ok=False)
        return
    try:
        def cb(pct,msg):
            set_job("dna",running=True,progress=int(pct),message=msg,ok=None)
        set_job("dna",running=True,progress=1,message="启动 Winner vs Loser DNA…",ok=None,result=None)
        result = run_winner_loser_diagnostic(str(data_path), str(BASE/"config.json"), progress_cb=cb)
        set_job("dna",running=False,progress=100,message="✅ V0.5 DNA 完成。下一步只验证最有分离度的单一入场前特征。",ok=True,result=result)
    except Exception as e:
        set_job("dna",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-dna")
def start_dna():
    with LOCK:
        if JOBS["dna"]["running"]:
            return jsonify(JOBS["dna"])
    threading.Thread(target=dna_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["dna"])


def entryq_worker():
    trades_path = OUT / "winner_loser_trades_v05.csv"
    if not trades_path.exists():
        set_job(
            "entryq", running=False, progress=100,
            message="❌ 找不到 V0.5 DNA Trade Cache。先跑一次 Winner vs Loser DNA；无需重新下载数据。",
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("entryq", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "entryq", running=True, progress=1,
            message="读取 /data/daytrade/outputs 的 V0.5 DNA Cache…",
            ok=None, result=None
        )
        result = run_entry_quality_validation(
            str(trades_path),
            str(BASE / "config.json"),
            outdir=str(OUT),
            progress_cb=cb
        )
        best = result.get("best_candidate")
        if best:
            msg = f"✅ V0.6 完成：{best['name']} = {best['verdict']}。先看 Retention / PF / AvgR / Stop / MaxDD / H1-H2，再决定是否晋级。"
        else:
            msg = "✅ V0.6 完成：没有 Gate 达到 PASS/WATCH。保留 Frozen Baseline，不硬调参数。"
        set_job("entryq", running=False, progress=100, message=msg, ok=True, result=result)
    except Exception as e:
        set_job("entryq", running=False, progress=100, message=f"❌ {type(e).__name__}: {e}", ok=False)


@app.post("/start-entryq")
def start_entryq():
    with LOCK:
        if JOBS["entryq"]["running"]:
            return jsonify(JOBS["entryq"])
    threading.Thread(target=entryq_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["entryq"])



def robust_worker():
    trades_path = OUT / "winner_loser_trades_v05.csv"
    if not trades_path.exists():
        set_job(
            "robust", running=False, progress=100,
            message="❌ 找不到 V0.5 DNA Trade Cache。只需先跑一次 Winner vs Loser DNA；不要重新下载数据。",
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("robust", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "robust", running=True, progress=1,
            message="读取 /data/daytrade/outputs 的 Frozen Trade Cache…",
            ok=None, result=None
        )
        result = run_robustness_v07(
            str(trades_path),
            outdir=str(OUT),
            progress_cb=cb
        )
        set_job(
            "robust", running=False, progress=100,
            message=f"✅ V0.7 完成：{result['verdict']} · {result['checks_passed']}/{result['checks_total']} checks",
            ok=True, result=result
        )
    except Exception as e:
        set_job(
            "robust", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}",
            ok=False
        )


@app.post("/start-robust")
def start_robust():
    with LOCK:
        if JOBS["robust"]["running"]:
            return jsonify(JOBS["robust"])
    threading.Thread(target=robust_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["robust"])



def oos_download_worker():
    try:
        OOS_DIR.mkdir(parents=True, exist_ok=True)
        OOS_CACHE.mkdir(parents=True, exist_ok=True)

        set_job("oosdl", running=True, progress=2, message="准备下载 untouched OOS 2024-08-30 → 2025-08-29", ok=None)

        import subprocess
        cmd = [
            sys.executable, "prepare_sip_data.py",
            "--start", "2024-08-30",
            "--end", "2025-08-29",
            "--freq", "5min",
            "--out", str(OOS_DATA),
            "--cache-dir", str(OOS_CACHE),
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        last = ""
        done_count = 0
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            last = line
            m = re.search(r"\\[(\\d+)\\s*/\\s*125\\]", line)
            if m:
                done_count = max(done_count, int(m.group(1)))
                pct = min(95, 5 + int(done_count / 125 * 90))
                set_job("oosdl", running=True, progress=pct, message=line, ok=None)

        rc = p.wait()
        if rc != 0:
            raise RuntimeError(last or f"prepare_sip_data.py exited {rc}")

        if not OOS_DATA.exists() or OOS_DATA.stat().st_size == 0:
            raise RuntimeError("OOS download finished but output file is missing/empty")

        set_job(
            "oosdl", running=False, progress=100,
            message=f"✅ Untouched OOS 下载完成：{round(OOS_DATA.stat().st_size/1024/1024,1)} MB，已写入 Persistent Volume",
            ok=True
        )
    except Exception as e:
        set_job("oosdl", running=False, progress=100, message=f"❌ {type(e).__name__}: {e}", ok=False)


@app.post("/start-oos-download")
def start_oos_download():
    with LOCK:
        if JOBS["oosdl"]["running"]:
            return jsonify(JOBS["oosdl"])
    threading.Thread(target=oos_download_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["oosdl"])


def oos_worker():
    oi = discover_oos_data(deep=True)
    if not oi["ready"]:
        set_job(
            "oos", running=False, progress=100,
            message=(
                "❌ Untouched OOS 尚未通过完整 Preflight："
                f"merged={oi['exists']} size={oi['size_mb']}MB "
                f"cache={oi['cache_files']}/125 rows={oi['rows']} symbols={oi['symbols']}。"
                "先 Run Preflight；不要重复下载，除非 Preflight 明确判定数据不完整。"
            ),
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("oos", running=True, progress=int(pct), message=msg, ok=None)

        set_job("oos", running=True, progress=1, message="读取 Persistent Untouched OOS…", ok=None, result=None)
        result = run_true_oos_v08(
            str(OOS_DATA),
            str(BASE / "config.json"),
            outdir=str(OOS_OUT),
            progress_cb=cb
        )
        set_job(
            "oos", running=False, progress=100,
            message=f"✅ V0.8 完成：{result['verdict']} · {result['checks_passed']}/{result['checks_total']} checks",
            ok=True, result=result
        )
    except Exception as e:
        set_job("oos", running=False, progress=100, message=f"❌ {type(e).__name__}: {e}", ok=False)


@app.post("/start-oos")
def start_oos():
    with LOCK:
        if JOBS["oos"]["running"]:
            return jsonify(JOBS["oos"])
    threading.Thread(target=oos_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["oos"])


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

    # Job state is in memory, but OOS data is persistent. Reconstruct a truthful
    # completed state after page refresh or Railway process restart.
    if kind == "oosdl" and not d.get("running"):
        oi = discover_oos_data(deep=False)
        if oi["ready"]:
            d.update({
                "progress": 100,
                "ok": True,
                "message": (
                    f"✅ Untouched OOS ready：{oi['size_mb']} MB · "
                    f"{oi['cache_files']}/125 cache files · Persistent Volume"
                ),
            })
        elif oi["exists"] or oi["cache_files"] > 0:
            d.update({
                "progress": min(99, int(oi["cache_files"] / 125 * 95) if oi["cache_files"] else 1),
                "ok": None,
                "message": (
                    f"发现 OOS 落盘数据：merged={oi['exists']} · "
                    f"cache {oi['cache_files']}/125。请 Run Preflight 做完整 Schema Audit。"
                ),
            })

    d["data_exists"]=DATA.exists() or SIP_DATA.exists()
    return jsonify(d)


@app.post("/rescan-data")
def rescan_data():
    info = discover_persistent_data()
    return jsonify({
        "ok": True,
        "replay_ready": info["replay_ready"],
        "sip_exists": info["sip_exists"],
        "sip_size_mb": info["sip_size_mb"],
        "cache_files": info["cache_files"],
        "cache_size_mb": info["cache_size_mb"],
        "data_root": info["data_root"],
    })

@app.get("/health")
def health():
    return {"status":"ok"},200

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT","8080")))
