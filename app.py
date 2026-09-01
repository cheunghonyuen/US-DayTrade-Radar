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
from oos_failure_v09 import run_oos_failure_autopsy
from setup_tournament_v1 import run_setup_tournament
from opening_drive_audit_v1 import run_opening_drive_audit
from opening_drive_decision_v1 import run_opening_drive_decision
from orb_frozen_audit_v1 import run_orb_frozen_audit
from first_pullback_frozen_audit_v1 import run_first_pullback_frozen_audit
from momentum_continuation_final_trial_v1 import run_momentum_continuation_final_trial
from exact_original_v01_replay import run_exact_original_v01
from original_engine_parity_audit_v1 import run_original_parity_audit
from original_v01_tzfixed_replay import run_original_tz_fixed_replay
from qc_stocks_in_play_benchmark_v1 import run_qc_stocks_in_play_benchmark
from qc_1min_oos_resolver_v1 import run_qc_1min_oos_resolver
from backtrader_execution_parity_v1 import run_backtrader_execution_parity
from backtrader_execution_parity_v2 import run_backtrader_parity_v2

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
    "autopsy": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "tournament": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "od_audit": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "od_decision": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "orb_audit": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "fp_audit": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "mc_trial": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "orig_v01": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "parity_audit": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "orig_tz": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "qc_bench": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "qc_1m": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "bt_parity": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
    "bt_parity_v2": {"running": False, "progress": 0, "message": "尚未开始", "ok": None, "result": None},
}

HTML = """
<!doctype html>
<html lang="zh">
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>US DayTrade Radar · Original Engine Parity Audit</title>
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
<h1>🇺🇸 US DayTrade Radar <small>Original Engine Parity Audit v1</small></h1>

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
<h2>10. V0.9 OOS Failure Autopsy</h2>
<p><b>研究对象：</b> V0.8 True OOS 已经失败的 Frozen Gate Trades</p>
<p class="muted">不再调 0.75% / 3.50%，不重新下载，也不重跑一年 Replay。直接读取 /data/daytrade/oos_v08/outputs/v08_oos_gate_trades.csv，比较 OOS Winner vs Loser，并检查失败是否集中在某些股票或月份。结果只做诊断，不在这个 OOS 年上拟合新阈值。</p>
<button id="autopsyBtn" type="button">Run V0.9 OOS Failure Autopsy</button>
<div class="row"><b>Autopsy 进度</b><span id="autopsyPct">0%</span></div>
<div class="track"><div id="autopsyBar" class="bar"></div></div>
<div id="autopsyMsg" class="msg">尚未开始</div>
<div id="autopsyResults" class="scroll"></div>
</div>

<div class="card">
<h2>11. Setup Tournament v1</h2>
<p><b>停止救 ORB。</b> 直接比较 4 个逻辑不同的美股日内 Setup。</p>
<p class="muted">复用现有 Discovery + True OOS 两年 Persistent Replay，不重新下载：ORB Breakout / Opening Drive / VWAP Reclaim / First Pullback。True OOS 优先，不按单一最高 PF 选冠军。</p>
<button id="tournamentBtn" type="button">Run Setup Tournament v1</button>
<div class="row"><b>Tournament 进度</b><span id="tournamentPct">0%</span></div>
<div class="track"><div id="tournamentBar" class="bar"></div></div>
<div id="tournamentMsg" class="msg">尚未开始</div>
<div id="tournamentResults" class="scroll"></div>
</div>

<div class="card">
<h2>12. Opening Drive Audit v1</h2>
<p><b>主线候选：</b> Opening Drive</p>
<p class="muted">不调参数。直接审计 Tournament 已保存的 Discovery + True OOS Opening Drive trades：月度稳定性、H1/H2、Winner Concentration、3000次 Bootstrap。</p>
<button id="od_auditBtn" type="button">Run Opening Drive Audit</button>
<div class="row"><b>Audit 进度</b><span id="od_auditPct">0%</span></div>
<div class="track"><div id="od_auditBar" class="bar"></div></div>
<div id="od_auditMsg" class="msg">尚未开始</div>
<div id="od_auditResults" class="scroll"></div>
</div>


<div class="card">
<h2>13. Opening Drive Decision v1</h2>
<p><b>去留裁决：</b> 不再救同一批样本上的 Opening Drive。</p>
<p class="muted">只读取已经完成的 Setup Tournament + Opening Drive Audit。AUDIT_PASS 才保留；否则从主线 DROP，并自动提升 Tournament 下一名。不会重新下载、不会重跑 Replay、不会调阈值。</p>
<button id="od_decisionBtn" type="button">Run Opening Drive Decision</button>
<div class="row"><b>Decision 进度</b><span id="od_decisionPct">0%</span></div>
<div class="track"><div id="od_decisionBar" class="bar"></div></div>
<div id="od_decisionMsg" class="msg">尚未开始</div>
<div id="od_decisionResults" class="scroll"></div>
</div>

<div class="card">
<h2>14. ORB Breakout Frozen Audit v1</h2>
<p><b>生死审判：</b>只审计 Tournament 已保存的 ORB Breakout trades。</p>
<p class="muted">不调 Gap/RVOL/OR 参数，不重跑历史、不重新下载。检查 True OOS PF、AvgR、MaxDD(R)、H1/H2、Winner Concentration、3000次 Bootstrap。只有 AUDIT_PASS 才进入 Paper Trading，否则 DROP。</p>
<button id="orb_auditBtn" type="button">Run ORB Frozen Audit</button>
<div class="row"><b>Audit 进度</b><span id="orb_auditPct">0%</span></div>
<div class="track"><div id="orb_auditBar" class="bar"></div></div>
<div id="orb_auditMsg" class="msg">尚未开始</div>
<div id="orb_auditResults" class="scroll"></div>
</div>

<div class="card">
<h2>15. First Pullback Frozen Audit v1</h2>
<p><b>生死审判：</b>只审计 Tournament 已保存的 First Pullback trades。</p>
<p class="muted">不调参数，不重新下载，不重跑 Replay。检查 True OOS PF、AvgR、MaxDD(R)、H1/H2、Winner Concentration、3000次 Bootstrap。只有 AUDIT_PASS 才进入 Paper Trading，否则 DROP。</p>
<button id="fp_auditBtn" type="button">Run First Pullback Frozen Audit</button>
<div class="row"><b>Audit 进度</b><span id="fp_auditPct">0%</span></div>
<div class="track"><div id="fp_auditBar" class="bar"></div></div>
<div id="fp_auditMsg" class="msg">尚未开始</div>
<div id="fp_auditResults" class="scroll"></div>
</div>

<div class="card">
<h2>16. Momentum Continuation Final Trial v1</h2>
<p><b>这一轮最后一个独立 Setup。</b></p>
<p><b>冻结规则：</b> Gap ≥2% / 前30分钟涨幅 ≥2% / 前30分钟 median RVOL ≥2.5 / 10:00时收盘位置位于前30分钟区间顶部25% / 10:00-11:30突破前30分钟高点并在 VWAP 上方 / Target 2R。</p>
<p class="muted">不扫描阈值、不重新下载。直接用现有 Discovery + True OOS 两年 Persistent Replay 做一次冻结测试。若失败，本轮 DayTrade Setup Discovery 结束，不再继续 V17/V18 救参数。</p>
<button id="mc_trialBtn" type="button">Run Momentum Continuation Final Trial</button>
<div class="row"><b>Trial 进度</b><span id="mc_trialPct">0%</span></div>
<div class="track"><div id="mc_trialBar" class="bar"></div></div>
<div id="mc_trialMsg" class="msg">尚未开始</div>
<div id="mc_trialResults" class="scroll"></div>
</div>

<div class="card">
<h2>17. Exact Original MVP v0.1 Replay</h2>
<p><b>这次不改模型。</b> 直接使用我们保存的 <code>US_DayTrade_MVP_v0.1.zip</code> 里的原始 <code>src/engine.py</code> + 原始 <code>config.json</code>。</p>
<p class="muted">原始规则：Gap 3%-20% / RVOL ≥1.2 / OR20 / 10:00-11:30 entry / 0.5×OR stop / 1.5R target / 15:50 EOD / $0.01 slippage / $0.005 commission。没有 VWAP、DNA、Filter Ladder、Frozen Gate、Tournament 或后来任何改动。</p>
<p class="muted">直接在现在同一批 125 Stocks-in-Play 的 Discovery + True OOS 两个时间段上跑，正面对比泛化能力。</p>
<button id="orig_v01Btn" type="button">Run Exact Original MVP v0.1</button>
<div class="row"><b>Original Replay 进度</b><span id="orig_v01Pct">0%</span></div>
<div class="track"><div id="orig_v01Bar" class="bar"></div></div>
<div id="orig_v01Msg" class="msg">尚未开始</div>
<div id="orig_v01Results" class="scroll"></div>
</div>

<div class="card">
<h2>18. Original Engine Parity Audit v1</h2>
<p><b>目的：</b>解释为什么 Exact Original MVP v0.1 出现 0 Trades。</p>
<p class="muted">不调任何模型参数。只检查输入兼容：timestamp 明确按 UTC 解析后转换到 America/New_York，再逐层统计 Regular Session → Prev Close → Gap 3%-20% → OR20 → Entry Window → Breakout → RVOL。</p>
<p class="muted">这一步不会重新下载数据，也不会重新优化模型。只要 funnel 恢复为非零，就证明前一次 0 Trades 属于 parity/time handling 问题，而不是模型表现结论。</p>
<button id="parity_auditBtn" type="button">Run Original Engine Parity Audit</button>
<div class="row"><b>Parity Audit 进度</b><span id="parity_auditPct">0%</span></div>
<div class="track"><div id="parity_auditBar" class="bar"></div></div>
<div id="parity_auditMsg" class="msg">尚未开始</div>
<div id="parity_auditResults" class="scroll"></div>
</div>

<div class="card">
<h2>19. Exact Original MVP v0.1 · TZ-Fixed Replay</h2>
<p><b>现在正式跑原版。</b> 交易规则完全不改，只修复 18 已确认的输入 parity：UTC timestamp → America/New_York → naive local wall-clock，再交给 preserved MVP v0.1 engine。</p>
<p class="muted">同一批 125 Stocks-in-Play；同一 Discovery + True OOS 时间段；原始 Gap 3%-20% / RVOL ≥1.2 / OR20 / 10:00-11:30 / 0.5×OR stop / 1.5R target / 15:50 EOD / 原始手续费与滑点。</p>
<p class="muted">这一步才是有效的 Original-vs-our-data 泛化测试。不会重新下载，不会改参数。</p>
<button id="orig_tzBtn" type="button">Run Exact Original · TZ-Fixed</button>
<div class="row"><b>Original Replay 进度</b><span id="orig_tzPct">0%</span></div>
<div class="track"><div id="orig_tzBar" class="bar"></div></div>
<div id="orig_tzMsg" class="msg">尚未开始</div>
<div id="orig_tzResults" class="scroll"></div>
</div>

<div class="card">
<h2>20. QuantConnect Stocks-in-Play ORB Benchmark v1</h2>
<p><b>外部高质量 Benchmark。</b> 不再修旧 MVP，也不扫参数。</p>
<p class="muted">按 QuantConnect/Zarattini ORB 公开规则移植：5min Opening Range、Price&gt;$5、ATR14&gt;$0.50、首5分钟 RVOL&gt;1、每天 Top20 RVOL、绿色开盘棒向上突破做多/红色向下突破做空、Stop=0.1×ATR、赢家拿到收盘。</p>
<p class="muted"><b>限制透明：</b>原研究 universe 是更大的高流动美股池；这里先严格限制在我们已有 125 Stocks-in-Play，不重新下载。因为我们只有5分钟数据，同时报告“同bar stop-first”与“下一bar stop生效”两个边界，避免用一个随意的成交顺序制造漂亮结果。</p>
<button id="qc_benchBtn" type="button">Run QuantConnect External Benchmark</button>
<div class="row"><b>Benchmark 进度</b><span id="qc_benchPct">0%</span></div>
<div class="track"><div id="qc_benchBar" class="bar"></div></div>
<div id="qc_benchMsg" class="msg">尚未开始</div>
<div id="qc_benchResults" class="scroll"></div>
</div>

<div class="card">
<h2>21. QuantConnect ORB · True OOS 1-Minute Resolver v1</h2>
<p><b>只解决真正会改变结论的问题：</b>20 的 OOS PF 从 0.6845 到 1.5743，差异来自同一根 5min bar 内 Entry/Stop 顺序。</p>
<p class="muted">这版只处理 True OOS，而且只下载“strict 与 delayed-stop 结果不同”的交易日 1-minute bars。已解析/下载的数据写入 /data/daytrade 持久缓存；Railway 中断后可继续，不从头下载。</p>
<p class="muted">1-minute 仍可能出现同一分钟同时触发 Entry+Stop，所以结果输出 Lower/Upper 两个边界。两边都盈利才晋级 Backtrader/LEAN；两边都亏就 DROP；仍跨过 PF=1 才需要 tick/event-order 数据。</p>
<button id="qc_1mBtn" type="button">Run True OOS 1-Minute Resolver</button>
<div class="row"><b>Resolver 进度</b><span id="qc_1mPct">0%</span></div>
<div class="track"><div id="qc_1mBar" class="bar"></div></div>
<div id="qc_1mMsg" class="msg">尚未开始</div>
<div id="qc_1mResults" class="scroll"></div>
</div>

<div class="card">
<h2>22. Backtrader Independent Engine Parity v1</h2>
<p><b>只验证引擎，不改策略。</b> 20 的选股/入场/止损全部冻结；21 已把 5min 歧义缩到 1min。</p>
<p class="muted">这一步使用独立的 Backtrader event-driven broker，只重放 21 的 161 笔真正有 execution ambiguity 的 True OOS 交易；其余 1652 笔 strict/delayed 已一致，不浪费时间重复跑。直接复用 /data/daytrade/qc_1min_oos_cache，不重新下载。</p>
<p class="muted">PASS 条件：Backtrader 汇总后的 True OOS PF&gt;1、AvgR&gt;0，并与 21 的正收益 Lower/Upper 边界一致。失败才查 execution；绝不调策略参数。</p>
<button id="bt_parityBtn" type="button">Run Backtrader Engine Parity</button>
<div class="row"><b>Parity 进度</b><span id="bt_parityPct">0%</span></div>
<div class="track"><div id="bt_parityBar" class="bar"></div></div>
<div id="bt_parityMsg" class="msg">尚未开始</div>
<div id="bt_parityResults" class="scroll"></div>
</div>

<div class="card">
<h2>23. Backtrader Parity · Runtime/Schema Fix v2</h2>
<p><b>只修22的执行适配，不碰策略。</b> 22 的 161/161 全失败，所以 PF 0.6845 无效。</p>
<p class="muted">这版先拿真实 /data/daytrade/qc_1min_oos_cache 做 Runtime + Schema Preflight，并实际用前5笔交易启动 Backtrader。Preflight 有任何错误就立即停，不再浪费时间长跑；通过才重放全部161笔。</p>
<p class="muted">改动仅限：明确1-share订单、真实1min PandasData schema、入场前一根bar提交stop-entry、订单状态日志。20/21的选股、Entry、Stop、Exit规则和参数全部冻结。</p>
<button id="bt_parity_v2Btn" type="button">Run Backtrader Parity v2</button>
<div class="row"><b>Parity v2 进度</b><span id="bt_parity_v2Pct">0%</span></div>
<div class="track"><div id="bt_parity_v2Bar" class="bar"></div></div>
<div id="bt_parity_v2Msg" class="msg">尚未开始</div>
<div id="bt_parity_v2Results" class="scroll"></div>
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


function renderAutopsy(r){
  if(!r) return;
  const s=r.gate_summary||{};
  let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b></p>`;
  h+=`<p><b>Gate OOS:</b> ${s.trades||0} trades · Win ${(Number(s.win_rate||0)*100).toFixed(1)}% · PF ${s.profit_factor||0} · AvgR ${s.avg_r||0} · Stop ${(Number(s.stop_rate||0)*100).toFixed(1)}%</p>`;
  h+=`<p>Winner ${r.winner_count} / Loser ${r.loser_count} · Target ${r.target_count} / Stop ${r.stop_count}</p>`;

  h+='<h3>Winner vs Loser Pre-entry DNA</h3><table><thead><tr><th>Feature</th><th>Winner Med</th><th>Loser Med</th><th>Delta</th><th>Effect</th><th>Direction</th></tr></thead><tbody>';
  (r.features||[]).forEach(x=>{
    h+=`<tr><td>${x.feature}</td><td>${x.winner_median}</td><td>${x.loser_median}</td><td>${x.delta}</td><td>${x.robust_effect}</td><td>${x.direction}</td></tr>`;
  });
  h+='</tbody></table>';

  h+='<h3>Worst Symbols</h3><table><thead><tr><th>Symbol</th><th>Trades</th><th>AvgR</th><th>Stop%</th></tr></thead><tbody>';
  (r.worst_symbols||[]).forEach(x=>{
    h+=`<tr><td>${x.symbol}</td><td>${x.trades}</td><td>${x.avg_r}</td><td>${(Number(x.stop_rate||0)*100).toFixed(1)}</td></tr>`;
  });
  h+='</tbody></table>';

  h+='<h3>Monthly Failure Map</h3><table><thead><tr><th>Month</th><th>Trades</th><th>Win%</th><th>AvgR</th><th>Stop%</th></tr></thead><tbody>';
  (r.monthly_failure_map||[]).forEach(x=>{
    h+=`<tr><td>${x.month}</td><td>${x.trades}</td><td>${(Number(x.win_rate||0)*100).toFixed(1)}</td><td>${x.avg_r}</td><td>${(Number(x.stop_rate||0)*100).toFixed(1)}</td></tr>`;
  });
  h+='</tbody></table>';

  h+=`<p><b>Next:</b> ${r.next_step}</p><p class="muted">${r.guardrail}</p>`;
  $("autopsyResults").innerHTML=h;
}


function renderTournament(r){
  if(!r)return;
  let h=`<p><b>${r.study}</b></p><p>${r.ranking_rule}</p>`;
  h+='<table><thead><tr><th>Setup</th><th>D Trades</th><th>D PF</th><th>D AvgR</th><th>OOS Trades</th><th>OOS PF</th><th>OOS AvgR</th><th>OOS Win%</th><th>Score</th><th>Verdict</th></tr></thead><tbody>';
  (r.results||[]).forEach(x=>{const d=x.discovery||{},o=x.oos||{};h+=`<tr><td><b>${x.setup}</b></td><td>${d.trades}</td><td>${d.profit_factor}</td><td>${d.avg_r}</td><td>${o.trades}</td><td>${o.profit_factor}</td><td>${o.avg_r}</td><td>${(Number(o.win_rate||0)*100).toFixed(1)}</td><td>${x.score}</td><td><b>${x.verdict}</b></td></tr>`;});
  h+='</tbody></table>';
  if(r.winner)h+=`<p><b>Current Leader:</b> ${r.winner.setup} · ${r.winner.verdict}</p>`;
  $("tournamentResults").innerHTML=h;
}

function renderODAudit(r){
 if(!r)return; const d=r.discovery||{},o=r.oos||{},c=r.oos_winner_concentration||{},b=r.oos_bootstrap||{};
 let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total}</p>`;
 h+=`<p><b>Discovery:</b> ${d.trades} trades · PF ${d.profit_factor} · AvgR ${d.avg_r}</p>`;
 h+=`<p><b>True OOS:</b> ${o.trades} trades · PF ${o.profit_factor} · AvgR ${o.avg_r} · Win ${(Number(o.win_rate||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>OOS Winner Concentration:</b> Top1 ${(Number(c.top1_share||0)*100).toFixed(1)}% · Top3 ${(Number(c.top3_share||0)*100).toFixed(1)}% · Top5 ${(Number(c.top5_share||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>OOS Bootstrap:</b> AvgR CI [${(b.avg_r_ci95||[]).join(", ")}] · PF CI [${(b.pf_ci95||[]).join(", ")}] · P(PF>1) ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}%</p>`;
 h+='<h3>OOS Half Split</h3><table><thead><tr><th>Half</th><th>Trades</th><th>Win%</th><th>PF</th><th>AvgR</th></tr></thead><tbody>';
 (r.oos_halves||[]).forEach(x=>h+=`<tr><td>${x.half}</td><td>${x.trades}</td><td>${(Number(x.win_rate||0)*100).toFixed(1)}</td><td>${x.profit_factor}</td><td>${x.avg_r}</td></tr>`);
 h+='</tbody></table><h3>Checks</h3><table><tbody>';
 Object.entries(r.checks||{}).forEach(([k,v])=>h+=`<tr><td>${k}</td><td><b>${v?"✅ PASS":"❌ FAIL"}</b></td></tr>`);
 h+='</tbody></table>';
 $("od_auditResults").innerHTML=h;
}

function renderODDecision(r){
  if(!r)return;
  let h=`<p><b>${r.study}</b></p>`;
  h+=`<p><b>Decision:</b> ${r.decision}</p>`;
  h+=`<p><b>Audit:</b> ${r.audit_verdict} · ${r.audit_checks}</p>`;
  h+=`<p><b>Failed checks:</b> ${(r.failed_checks||[]).join(", ") || "None"}</p>`;
  h+=`<p><b>Next Mainline:</b> ${r.next_mainline || "NONE"}</p>`;
  if(r.next_candidate){
    const d=r.next_candidate.discovery||{}, o=r.next_candidate.oos||{};
    h+=`<p><b>Promoted candidate:</b> ${r.next_candidate.setup} · ${r.next_candidate.verdict} · D PF ${d.profit_factor} / OOS PF ${o.profit_factor}</p>`;
  }
  h+=`<p>${r.reason}</p><p class="muted">${r.rule}</p>`;
  $("od_decisionResults").innerHTML=h;
}


function renderORBAudit(r){
 if(!r)return; const d=r.discovery||{},o=r.oos||{},c=r.oos_winner_concentration||{},b=r.oos_bootstrap||{};
 let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total}</p>`;
 h+=`<p><b>Decision:</b> ${r.decision}</p>`;
 h+=`<p><b>Discovery:</b> ${d.trades} trades · PF ${d.profit_factor} · AvgR ${d.avg_r} · MaxDD ${d.max_dd_r}R</p>`;
 h+=`<p><b>True OOS:</b> ${o.trades} trades · PF ${o.profit_factor} · AvgR ${o.avg_r} · MaxDD ${o.max_dd_r}R · Win ${(Number(o.win_rate||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>OOS Top5 winner share:</b> ${(Number(c.top5_share||0)*100).toFixed(1)}% · <b>P(PF>1):</b> ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}%</p>`;
 h+='<h3>OOS H1/H2</h3><table><thead><tr><th>Half</th><th>Trades</th><th>PF</th><th>AvgR</th><th>MaxDD(R)</th></tr></thead><tbody>';
 (r.oos_halves||[]).forEach(x=>h+=`<tr><td>${x.half}</td><td>${x.trades}</td><td>${x.profit_factor}</td><td>${x.avg_r}</td><td>${x.max_dd_r}</td></tr>`);
 h+='</tbody></table><h3>Checks</h3><table><tbody>';
 Object.entries(r.checks||{}).forEach(([k,v])=>h+=`<tr><td>${k}</td><td><b>${v?"✅ PASS":"❌ FAIL"}</b></td></tr>`);
 h+='</tbody></table>';
 $("orb_auditResults").innerHTML=h;
}


function renderFPAudit(r){
 if(!r)return;
 const d=r.discovery||{},o=r.oos||{},c=r.oos_winner_concentration||{},b=r.oos_bootstrap||{};
 let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total}</p>`;
 h+=`<p><b>Decision:</b> ${r.decision}</p>`;
 h+=`<p><b>Discovery:</b> ${d.trades} trades · PF ${d.profit_factor} · AvgR ${d.avg_r} · MaxDD ${d.max_dd_r}R</p>`;
 h+=`<p><b>True OOS:</b> ${o.trades} trades · PF ${o.profit_factor} · AvgR ${o.avg_r} · MaxDD ${o.max_dd_r}R · Win ${(Number(o.win_rate||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>OOS Top5 winner share:</b> ${(Number(c.top5_share||0)*100).toFixed(1)}% · <b>P(PF>1):</b> ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}%</p>`;
 h+='<h3>OOS H1/H2</h3><table><thead><tr><th>Half</th><th>Trades</th><th>PF</th><th>AvgR</th><th>MaxDD(R)</th></tr></thead><tbody>';
 (r.oos_halves||[]).forEach(x=>h+=`<tr><td>${x.half}</td><td>${x.trades}</td><td>${x.profit_factor}</td><td>${x.avg_r}</td><td>${x.max_dd_r}</td></tr>`);
 h+='</tbody></table><h3>Checks</h3><table><tbody>';
 Object.entries(r.checks||{}).forEach(([k,v])=>h+=`<tr><td>${k}</td><td><b>${v?"✅ PASS":"❌ FAIL"}</b></td></tr>`);
 h+='</tbody></table>';
 $("fp_auditResults").innerHTML=h;
}


function renderMCTrial(r){
 if(!r)return;
 const d=r.discovery||{},o=r.oos||{},c=r.oos_winner_concentration||{},b=r.oos_bootstrap||{};
 let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b> · ${r.checks_passed}/${r.checks_total}</p>`;
 h+=`<p><b>Decision:</b> ${r.decision}</p>`;
 h+=`<p><b>Discovery:</b> ${d.trades} trades · PF ${d.profit_factor} · AvgR ${d.avg_r} · MaxDD ${d.max_dd_r}R</p>`;
 h+=`<p><b>True OOS:</b> ${o.trades} trades · PF ${o.profit_factor} · AvgR ${o.avg_r} · MaxDD ${o.max_dd_r}R · Win ${(Number(o.win_rate||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>OOS Top5 winner share:</b> ${(Number(c.top5_share||0)*100).toFixed(1)}% · <b>P(PF>1):</b> ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}%</p>`;
 h+='<h3>OOS H1/H2</h3><table><thead><tr><th>Half</th><th>Trades</th><th>PF</th><th>AvgR</th><th>MaxDD(R)</th></tr></thead><tbody>';
 (r.oos_halves||[]).forEach(x=>h+=`<tr><td>${x.half}</td><td>${x.trades}</td><td>${x.profit_factor}</td><td>${x.avg_r}</td><td>${x.max_dd_r}</td></tr>`);
 h+='</tbody></table><h3>Checks</h3><table><tbody>';
 Object.entries(r.checks||{}).forEach(([k,v])=>h+=`<tr><td>${k}</td><td><b>${v?"✅ PASS":"❌ FAIL"}</b></td></tr>`);
 h+='</tbody></table>';
 $("mc_trialResults").innerHTML=h;
}


function renderOrigV01(r){
 if(!r)return;
 const d=r.discovery||{},o=r.true_oos||{};
 const pct=x=>(Number(x||0)*100).toFixed(1);
 let h=`<p><b>${r.study}</b></p><p><b>Source:</b> ${r.source}</p>`;
 h+=`<p><b>Mode:</b> ${r.engine_mode}</p>`;
 h+='<table><thead><tr><th>Period</th><th>Trades</th><th>Win%</th><th>PF</th><th>Avg R</th><th>MaxDD%</th><th>Net PnL</th><th>Stop%</th><th>Target%</th></tr></thead><tbody>';
 h+=`<tr><td>Discovery</td><td>${d.trades??"—"}</td><td>${pct(d.win_rate)}</td><td>${d.profit_factor??"—"}</td><td>${d.avg_r??"—"}</td><td>${d.max_drawdown_pct??"—"}</td><td>${d.net_pnl??"—"}</td><td>${pct(d.stop_rate)}</td><td>${pct(d.target_rate)}</td></tr>`;
 h+=`<tr><td>True OOS</td><td>${o.trades??"—"}</td><td>${pct(o.win_rate)}</td><td>${o.profit_factor??"—"}</td><td>${o.avg_r??"—"}</td><td>${o.max_drawdown_pct??"—"}</td><td>${o.net_pnl??"—"}</td><td>${pct(o.stop_rate)}</td><td>${pct(o.target_rate)}</td></tr>`;
 h+='</tbody></table>';
 h+=`<p>${r.comparison_note}</p><p><b>Interpretation:</b> ${r.interpretation}</p>`;
 $("orig_v01Results").innerHTML=h;
}


function renderParityAudit(r){
 if(!r)return;
 const d=r.discovery||{},o=r.true_oos||{};
 const df=d.funnel||{},of=o.funnel||{};
 let h=`<p><b>${r.study}</b></p>`;
 h+=`<p><b>Timezone:</b> ${r.timezone_assumption}</p>`;
 h+='<table><thead><tr><th>Stage</th><th>Discovery</th><th>True OOS</th></tr></thead><tbody>';
 const rows=[
  ["Raw rows",d.raw_rows,o.raw_rows],
  ["Regular session rows",d.regular_session_rows_after_tz_fix,o.regular_session_rows_after_tz_fix],
  ["Sessions total",df.sessions_total,of.sessions_total],
  ["Prev close",df.prev_close_available,of.prev_close_available],
  ["Gap 3%-20%",df.gap_3_to_20_pass,of.gap_3_to_20_pass],
  ["OR20 valid",df.or20_valid,of.or20_valid],
  ["Entry window bars",df.entry_window_has_bars,of.entry_window_has_bars],
  ["Breakout",df.breakout_pass,of.breakout_pass],
  ["RVOL diagnostic",df.rvol_diag_pass,of.rvol_diag_pass]
 ];
 rows.forEach(x=>h+=`<tr><td>${x[0]}</td><td>${x[1]??"—"}</td><td>${x[2]??"—"}</td></tr>`);
 h+='</tbody></table>';
 h+=`<p><b>Discovery NY time:</b> ${(d.time_distribution||{}).min_time||"—"} → ${(d.time_distribution||{}).max_time||"—"}</p>`;
 h+=`<p><b>True OOS NY time:</b> ${(o.time_distribution||{}).min_time||"—"} → ${(o.time_distribution||{}).max_time||"—"}</p>`;
 h+=`<p>${r.decision_rule}</p>`;
 $("parity_auditResults").innerHTML=h;
}


function renderOrigTZ(r){
 if(!r)return;
 const d=r.discovery||{},o=r.true_oos||{},p=r.parity||{};
 const pct=x=>(Number(x||0)*100).toFixed(1);
 let h=`<p><b>${r.study}</b></p>`;
 h+=`<p><b>Rule changes:</b> ${r.rule_changes}</p>`;
 h+=`<p><b>Parity fix:</b> ${r.input_parity_fix}</p>`;
 h+='<table><thead><tr><th>Period</th><th>Trades</th><th>Win%</th><th>PF</th><th>Avg R</th><th>MaxDD%</th><th>Net PnL</th><th>Stop%</th><th>Target%</th></tr></thead><tbody>';
 h+=`<tr><td>Discovery</td><td>${d.trades??"—"}</td><td>${pct(d.win_rate)}</td><td>${d.profit_factor??"—"}</td><td>${d.avg_r??"—"}</td><td>${d.max_drawdown_pct??"—"}</td><td>${d.net_pnl??"—"}</td><td>${pct(d.stop_rate)}</td><td>${pct(d.target_rate)}</td></tr>`;
 h+=`<tr><td>True OOS</td><td>${o.trades??"—"}</td><td>${pct(o.win_rate)}</td><td>${o.profit_factor??"—"}</td><td>${o.avg_r??"—"}</td><td>${o.max_drawdown_pct??"—"}</td><td>${o.net_pnl??"—"}</td><td>${pct(o.stop_rate)}</td><td>${pct(o.target_rate)}</td></tr>`;
 h+='</tbody></table>';
 if(p.discovery && p.true_oos){
   h+=`<p><b>Parity:</b> Discovery NY ${p.discovery.ny_time_min}→${p.discovery.ny_time_max}, trades ${p.discovery.trades}; True OOS NY ${p.true_oos.ny_time_min}→${p.true_oos.ny_time_max}, trades ${p.true_oos.trades}</p>`;
 }
 h+=`<p><b>Decision rule:</b> ${r.decision_rule}</p>`;
 $("orig_tzResults").innerHTML=h;
}


function renderQCBench(r){
 if(!r)return;
 const d=r.discovery||{},o=r.true_oos||{};
 const ds=d.strict||{}, dd=d.delayed_stop||{}, os=o.strict||{}, od=o.delayed_stop||{};
 let h=`<p><b>${r.study}</b></p>`;
 h+=`<p><b>Constraint:</b> ${r.fidelity.known_constraint}</p>`;
 h+='<table><thead><tr><th>Period / Fill bound</th><th>Trades</th><th>Win%</th><th>PF</th><th>AvgR</th><th>MedianR</th><th>MaxDD(R)</th><th>Stop%</th></tr></thead><tbody>';
 const row=(name,x)=>`<tr><td>${name}</td><td>${x.trades??0}</td><td>${(Number(x.win_rate||0)*100).toFixed(1)}</td><td>${x.profit_factor??0}</td><td>${x.avg_r??0}</td><td>${x.median_r??0}</td><td>${x.max_dd_r??0}</td><td>${(Number(x.stop_rate||0)*100).toFixed(1)}</td></tr>`;
 h+=row("Discovery · strict",ds)+row("Discovery · delayed-stop",dd)+row("True OOS · strict",os)+row("True OOS · delayed-stop",od);
 h+='</tbody></table>';
 const b=o.bootstrap_strict||{};
 h+=`<p><b>OOS strict bootstrap:</b> P(PF&gt;1) ${(Number(b.prob_pf_gt1||0)*100).toFixed(1)}% · PF95% [${(b.pf_ci95||[]).join(", ")}] · P(AvgR&gt;0) ${(Number(b.prob_avg_r_gt0||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>Meta:</b> Discovery candidates ${d.meta?.eligible_candidates??0}; OOS candidates ${o.meta?.eligible_candidates??0}; symbols ${o.meta?.symbols??0}</p>`;
 h+=`<p><b>Decision:</b> ${r.decision_rule}</p>`;
 $("qc_benchResults").innerHTML=h;
}


function renderQC1m(r){
 if(!r)return;
 const lo=r.lower_bound||{},hi=r.upper_bound||{};
 let h=`<p><b>${r.study}</b></p>`;
 h+=`<p>OOS trades ${r.oos_trades} · 5min ambiguous ${r.five_min_ambiguous_trades} · stable ${r.five_min_stable_trades}</p>`;
 h+=`<p><b>Resolution:</b> ${JSON.stringify(r.resolution_status||{})}</p>`;
 h+='<table><thead><tr><th>1min bound</th><th>Trades</th><th>Win%</th><th>PF</th><th>AvgR</th><th>MedianR</th></tr></thead><tbody>';
 const row=(n,x)=>`<tr><td>${n}</td><td>${x.trades??0}</td><td>${(Number(x.win_rate||0)*100).toFixed(1)}</td><td>${x.profit_factor??0}</td><td>${x.avg_r??0}</td><td>${x.median_r??0}</td></tr>`;
 h+=row("Lower (conservative)",lo)+row("Upper",hi);
 h+='</tbody></table>';
 h+=`<p><b>Remaining same-1min ambiguity:</b> ${r.remaining_1min_ambiguity} · <b>Download errors:</b> ${r.download_errors}</p>`;
 h+=`<p><b>Decision:</b> ${r.decision_rule}</p>`;
 $("qc_1mResults").innerHTML=h;
}


function renderBTParity(r){
 if(!r)return;
 const lo=r.lower_bound_21||{}, hi=r.upper_bound_21||{}, b=r.backtrader_oos||{};
 const pct=x=>(Number(x||0)*100).toFixed(1);
 let h=`<p><b>${r.study}</b> · <b>${r.parity_pass?"✅ PASS":"❌ FAIL"}</b></p>`;
 h+=`<p><b>Engine:</b> ${r.engine} ${r.backtrader_version} · Ambiguous ${r.ambiguous_trades} · resolved ${r.backtrader_resolved} · errors ${r.backtrader_errors}</p>`;
 h+='<table><thead><tr><th>Result</th><th>Trades</th><th>Win%</th><th>PF</th><th>AvgR</th><th>MedianR</th></tr></thead><tbody>';
 const row=(n,x)=>`<tr><td>${n}</td><td>${x.trades??0}</td><td>${pct(x.win_rate)}</td><td>${x.profit_factor??0}</td><td>${x.avg_r??0}</td><td>${x.median_r??0}</td></tr>`;
 h+=row("21 Lower",lo)+row("Backtrader",b)+row("21 Upper",hi);
 h+='</tbody></table>';
 h+=`<p><b>Ambiguous trades inside 21 bounds:</b> ${(Number(r.ambiguous_trade_result_inside_21_bounds_share||0)*100).toFixed(1)}%</p>`;
 h+=`<p><b>Decision:</b> ${r.decision_rule}</p>`;
 $("bt_parityResults").innerHTML=h;
}


function renderBTParityV2(r){
 if(!r)return;
 let h=`<p><b>${r.study}</b> · <b>${r.verdict}</b></p>`;
 if(r.preflight){
   h+=`<p><b>Preflight:</b> Backtrader ${r.preflight.backtrader_version} · cache files ${r.preflight.cache_files_total} · ambiguous ${r.preflight.ambiguous}</p>`;
   h+='<table><thead><tr><th>Sample</th><th>Exists</th><th>Rows</th><th>Trial</th><th>Error</th></tr></thead><tbody>';
   (r.preflight.sample_checks||[]).forEach(x=>h+=`<tr><td>${x.symbol} ${x.date}</td><td>${x.exists?"✅":"❌"}</td><td>${x.rows??0}</td><td>${x.trial_status??"—"}</td><td>${x.trial_error??""}</td></tr>`);
   h+='</tbody></table>';
 }
 if(r.verdict==="PREFLIGHT_FAIL"){
   h+=`<p><b>STOP：</b>${r.note}</p>`;
   $("bt_parity_v2Results").innerHTML=h; return;
 }
 const lo=r.lower_bound_21||{}, b=r.backtrader_oos||{}, hi=r.upper_bound_21||{};
 const pct=x=>(Number(x||0)*100).toFixed(1);
 h+=`<p><b>Resolved:</b> ${r.resolved}/${r.ambiguous_trades} · <b>Errors:</b> ${r.errors} · <b>Inside 21 bounds:</b> ${(Number(r.inside_21_bounds_share||0)*100).toFixed(1)}%</p>`;
 h+='<table><thead><tr><th>Result</th><th>Trades</th><th>Win%</th><th>PF</th><th>AvgR</th><th>MedianR</th></tr></thead><tbody>';
 const row=(n,x)=>`<tr><td>${n}</td><td>${x.trades??0}</td><td>${pct(x.win_rate)}</td><td>${x.profit_factor??0}</td><td>${x.avg_r??0}</td><td>${x.median_r??0}</td></tr>`;
 h+=row("21 Lower",lo)+row("Backtrader v2",b)+row("21 Upper",hi);
 h+='</tbody></table><p><b>Decision:</b> '+(r.decision_rule||"")+'</p>';
 $("bt_parity_v2Results").innerHTML=h;
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
    if(d.result && kind==="autopsy") renderAutopsy(d.result);
    if(d.result && kind==="tournament") renderTournament(d.result);
    if(d.result && kind==="od_audit") renderODAudit(d.result);
    if(d.result && kind==="od_decision") renderODDecision(d.result);
    if(d.result && kind==="orb_audit") renderORBAudit(d.result);
    if(d.result && kind==="fp_audit") renderFPAudit(d.result);
    if(d.result && kind==="mc_trial") renderMCTrial(d.result);
    if(d.result && kind==="orig_v01") renderOrigV01(d.result);
    if(d.result && kind==="parity_audit") renderParityAudit(d.result);
    if(d.result && kind==="orig_tz") renderOrigTZ(d.result);
    if(d.result && kind==="qc_bench") renderQCBench(d.result);
    if(d.result && kind==="qc_1m") renderQC1m(d.result);
    if(d.result && kind==="bt_parity") renderBTParity(d.result);
    if(d.result && kind==="bt_parity_v2") renderBTParityV2(d.result);
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


$("autopsyBtn").addEventListener("click", async ()=>{
  const d=await (await fetch("/start-autopsy",{method:"POST"})).json();
  setProgress("autopsy",d); poll("autopsy");
});

$("tournamentBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-tournament",{method:"POST"})).json(); setProgress("tournament",d); poll("tournament"); });

$("od_auditBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-od-audit",{method:"POST"})).json(); setProgress("od_audit",d); poll("od_audit"); });
$("od_decisionBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-od-decision",{method:"POST"})).json(); setProgress("od_decision",d); poll("od_decision"); poll("orb_audit"); poll("fp_audit"); poll("mc_trial"); poll("orig_v01"); poll("parity_audit"); poll("orig_tz"); poll("qc_bench"); poll("qc_1m"); poll("bt_parity"); poll("bt_parity_v2"); });
$("orb_auditBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-orb-audit",{method:"POST"})).json(); setProgress("orb_audit",d); poll("orb_audit"); });
$("fp_auditBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-fp-audit",{method:"POST"})).json(); setProgress("fp_audit",d); poll("fp_audit"); });
$("mc_trialBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-mc-trial",{method:"POST"})).json(); setProgress("mc_trial",d); poll("mc_trial"); });
$("orig_v01Btn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-orig-v01",{method:"POST"})).json(); setProgress("orig_v01",d); poll("orig_v01"); });
$("parity_auditBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-parity-audit",{method:"POST"})).json(); setProgress("parity_audit",d); poll("parity_audit"); });
$("orig_tzBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-orig-tz",{method:"POST"})).json(); setProgress("orig_tz",d); poll("orig_tz"); });
$("qc_benchBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-qc-bench",{method:"POST"})).json(); setProgress("qc_bench",d); poll("qc_bench"); });
$("qc_1mBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-qc-1m",{method:"POST"})).json(); setProgress("qc_1m",d); poll("qc_1m"); });
$("bt_parityBtn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-bt-parity",{method:"POST"})).json(); setProgress("bt_parity",d); poll("bt_parity"); });
$("bt_parity_v2Btn").addEventListener("click", async ()=>{ const d=await (await fetch("/start-bt-parity-v2",{method:"POST"})).json(); setProgress("bt_parity_v2",d); poll("bt_parity_v2"); });

$("rescanBtn").addEventListener("click", async ()=>{
  try{
    const d=await (await fetch("/rescan-data",{method:"POST"})).json();
    $("rescanMsg").textContent = `Replay ready: ${d.replay_ready ? "YES" : "NO"} | sip_data: ${d.sip_size_mb} MB | cache: ${d.cache_files} files / ${d.cache_size_mb} MB`;
    if(d.replay_ready) $("dataState").textContent="✅ Persistent Replay ready";
  }catch(e){
    $("rescanMsg").textContent="Rescan failed";
  }
});

poll("download"); poll("backtest"); poll("matrix"); poll("ladder"); poll("dna"); poll("entryq"); poll("robust"); poll("oosdl"); poll("oos"); poll("autopsy"); poll("tournament"); poll("od_audit"); poll("od_decision");
// Mobile dashboard: numbered research cards are always newest-first.
(()=>{const wrap=document.querySelector(".wrap"); if(!wrap)return; const cards=[...wrap.querySelectorAll(":scope > .card")];
const numbered=cards.map(c=>{const h=c.querySelector("h2");const m=h&&h.textContent.match(/^\s*(\d+)\./);return {c,n:m?Number(m[1]):-1};});
numbered.filter(x=>x.n>=0).sort((a,b)=>b.n-a.n).forEach(x=>wrap.appendChild(x.c));
numbered.filter(x=>x.n<0).forEach(x=>wrap.appendChild(x.c));})();
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
    for f in ["engine.py","run_backtest.py","prepare_tiingo_data.py","prepare_sip_data.py","filter_ladder.py","winner_loser_diag.py","entry_quality_v06.py","robustness_v07.py","true_oos_v08.py","oos_failure_v09.py","setup_tournament_v1.py","opening_drive_audit_v1.py","config.json"]:
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
        import numpy, requests, backtrader
        checks.append(f"OK python dependencies · backtrader {getattr(backtrader,'__version__','unknown')}")
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



def autopsy_worker():
    gate_path = OOS_OUT / "v08_oos_gate_trades.csv"
    base_path = OOS_OUT / "v08_oos_baseline_trades.csv"

    if not gate_path.exists():
        set_job(
            "autopsy", running=False, progress=100,
            message="❌ 找不到 V0.8 OOS Gate Trades。先跑一次 V0.8 True OOS；不要重新下载数据。",
            ok=False
        )
        return

    try:
        def cb(pct, msg):
            set_job("autopsy", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "autopsy", running=True, progress=1,
            message="读取 Persistent V0.8 OOS trade outputs…",
            ok=None, result=None
        )
        result = run_oos_failure_autopsy(
            str(gate_path),
            str(base_path) if base_path.exists() else None,
            outdir=str(OOS_OUT),
            progress_cb=cb
        )
        set_job(
            "autopsy", running=False, progress=100,
            message=f"✅ V0.9 完成：{result['verdict']}",
            ok=True, result=result
        )
    except Exception as e:
        set_job(
            "autopsy", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}",
            ok=False
        )


@app.post("/start-autopsy")
def start_autopsy():
    with LOCK:
        if JOBS["autopsy"]["running"]:
            return jsonify(JOBS["autopsy"])
    threading.Thread(target=autopsy_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["autopsy"])



def tournament_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job("tournament",running=False,progress=100,message="❌ 缺少现有两年 Persistent Replay。先 Run Preflight；不要重新下载。",ok=False)
        return
    try:
        def cb(pct,msg): set_job("tournament",running=True,progress=int(pct),message=msg,ok=None)
        set_job("tournament",running=True,progress=1,message="读取两年 Persistent Replay…",ok=None,result=None)
        result=run_setup_tournament(str(SIP_DATA),str(OOS_DATA),str(BASE/"config.json"),str(OUT/"setup_tournament_v1"),progress_cb=cb)
        w=result.get("winner"); msg="✅ Setup Tournament v1 完成"+(f"：Leader = {w['setup']} / {w['verdict']}" if w else "")
        set_job("tournament",running=False,progress=100,message=msg,ok=True,result=result)
    except Exception as e:
        set_job("tournament",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-tournament")
def start_tournament():
    with LOCK:
        if JOBS["tournament"]["running"]: return jsonify(JOBS["tournament"])
    threading.Thread(target=tournament_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:return jsonify(JOBS["tournament"])


def od_audit_worker():
    tdir=OUT/"setup_tournament_v1"
    dpath=tdir/"opening_drive_discovery.csv"
    opath=tdir/"opening_drive_oos.csv"
    if not dpath.exists() or not opath.exists():
        set_job("od_audit",running=False,progress=100,message="❌ 找不到 Opening Drive tournament trade files。先跑 Setup Tournament v1；不要重新下载数据。",ok=False); return
    try:
        def cb(pct,msg): set_job("od_audit",running=True,progress=int(pct),message=msg,ok=None)
        set_job("od_audit",running=True,progress=1,message="读取 Tournament Opening Drive trades…",ok=None,result=None)
        result=run_opening_drive_audit(str(dpath),str(opath),str(tdir/"opening_drive_audit_v1"),progress_cb=cb)
        set_job("od_audit",running=False,progress=100,message=f"✅ Opening Drive Audit 完成：{result['verdict']} · {result['checks_passed']}/{result['checks_total']}",ok=True,result=result)
    except Exception as e:
        set_job("od_audit",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-od-audit")
def start_od_audit():
    with LOCK:
        if JOBS["od_audit"]["running"]: return jsonify(JOBS["od_audit"])
    threading.Thread(target=od_audit_worker,daemon=True).start(); time.sleep(.05)
    with LOCK:return jsonify(JOBS["od_audit"])


def od_decision_worker():
    tdir = OUT / "setup_tournament_v1"
    tournament_json = tdir / "setup_tournament_v1.json"
    audit_json = tdir / "opening_drive_audit_v1" / "opening_drive_audit_v1.json"
    try:
        set_job("od_decision", running=True, progress=20,
                message="读取已保存 Tournament + Opening Drive Audit…", ok=None, result=None)
        result = run_opening_drive_decision(
            str(tournament_json), str(audit_json), str(tdir / "opening_drive_decision_v1")
        )
        set_job("od_decision", running=False, progress=100,
                message=f"✅ Decision 完成：{result['decision']} → {result.get('next_mainline')}",
                ok=True, result=result)
    except Exception as e:
        set_job("od_decision", running=False, progress=100,
                message=f"❌ {type(e).__name__}: {e}", ok=False)

@app.post("/start-od-decision")
def start_od_decision():
    with LOCK:
        if JOBS["od_decision"]["running"]:
            return jsonify(JOBS["od_decision"])
    threading.Thread(target=od_decision_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["od_decision"])



def orb_audit_worker():
    tdir=OUT/"setup_tournament_v1"
    dpath=tdir/"orb_breakout_discovery.csv"
    opath=tdir/"orb_breakout_oos.csv"
    if not dpath.exists() or not opath.exists():
        set_job("orb_audit",running=False,progress=100,message="❌ 找不到 Tournament ORB trade files。先跑 Setup Tournament v1；不要重新下载。",ok=False); return
    try:
        def cb(pct,msg): set_job("orb_audit",running=True,progress=int(pct),message=msg,ok=None)
        set_job("orb_audit",running=True,progress=2,message="读取 frozen ORB trades…",ok=None,result=None)
        result=run_orb_frozen_audit(str(dpath),str(opath),str(tdir/"orb_frozen_audit_v1"),progress_cb=cb)
        set_job("orb_audit",running=False,progress=100,message=f"✅ ORB Frozen Audit：{result['verdict']} → {result['decision']}",ok=True,result=result)
    except Exception as e:
        set_job("orb_audit",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-orb-audit")
def start_orb_audit():
    with LOCK:
        if JOBS["orb_audit"]["running"]: return jsonify(JOBS["orb_audit"])
    threading.Thread(target=orb_audit_worker,daemon=True).start(); time.sleep(.05)
    with LOCK:return jsonify(JOBS["orb_audit"])



def fp_audit_worker():
    tdir = OUT / "setup_tournament_v1"
    dpath = tdir / "first_pullback_discovery.csv"
    opath = tdir / "first_pullback_oos.csv"

    if not dpath.exists() or not opath.exists():
        set_job(
            "fp_audit", running=False, progress=100,
            message="❌ 找不到 Tournament First Pullback trade files。先跑 Setup Tournament v1；不要重新下载。",
            ok=False
        )
        return

    try:
        def cb(pct, msg):
            set_job("fp_audit", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "fp_audit", running=True, progress=2,
            message="读取 frozen First Pullback trades…", ok=None, result=None
        )

        result = run_first_pullback_frozen_audit(
            str(dpath), str(opath),
            str(tdir / "first_pullback_frozen_audit_v1"),
            progress_cb=cb
        )

        set_job(
            "fp_audit", running=False, progress=100,
            message=f"✅ First Pullback Frozen Audit：{result['verdict']} → {result['decision']}",
            ok=True, result=result
        )

    except Exception as e:
        set_job(
            "fp_audit", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}", ok=False
        )

@app.post("/start-fp-audit")
def start_fp_audit():
    with LOCK:
        if JOBS["fp_audit"]["running"]:
            return jsonify(JOBS["fp_audit"])
    threading.Thread(target=fp_audit_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["fp_audit"])



def mc_trial_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job(
            "mc_trial", running=False, progress=100,
            message="❌ 缺少现有 Discovery / True OOS Persistent Replay。先 Run Preflight；不要重新下载。",
            ok=False
        )
        return

    try:
        def cb(pct, msg):
            set_job("mc_trial", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "mc_trial", running=True, progress=2,
            message="读取两年 Persistent Replay；冻结规则，不扫描参数…",
            ok=None, result=None
        )

        result = run_momentum_continuation_final_trial(
            str(SIP_DATA), str(OOS_DATA), str(BASE / "config.json"),
            str(OUT / "momentum_continuation_final_trial_v1"),
            progress_cb=cb
        )

        set_job(
            "mc_trial", running=False, progress=100,
            message=f"✅ Momentum Continuation Final Trial：{result['verdict']} → {result['decision']}",
            ok=True, result=result
        )

    except Exception as e:
        set_job(
            "mc_trial", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}", ok=False
        )

@app.post("/start-mc-trial")
def start_mc_trial():
    with LOCK:
        if JOBS["mc_trial"]["running"]:
            return jsonify(JOBS["mc_trial"])
    threading.Thread(target=mc_trial_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["mc_trial"])



def orig_v01_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job(
            "orig_v01", running=False, progress=100,
            message="❌ 缺少现有 Discovery / True OOS Persistent Replay。不要重新下载；先 Run Preflight。",
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("orig_v01", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "orig_v01", running=True, progress=2,
            message="加载 preserved MVP v0.1 原始 engine + 原始 config；不使用后续改动…",
            ok=None, result=None
        )
        result = run_exact_original_v01(
            str(SIP_DATA), str(OOS_DATA),
            str(OUT / "exact_original_v01_replay"),
            progress_cb=cb
        )
        set_job(
            "orig_v01", running=False, progress=100,
            message="✅ Exact Original MVP v0.1 Replay 完成",
            ok=True, result=result
        )
    except Exception as e:
        set_job(
            "orig_v01", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}", ok=False
        )

@app.post("/start-orig-v01")
def start_orig_v01():
    with LOCK:
        if JOBS["orig_v01"]["running"]:
            return jsonify(JOBS["orig_v01"])
    threading.Thread(target=orig_v01_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["orig_v01"])



def parity_audit_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job(
            "parity_audit", running=False, progress=100,
            message="❌ 缺少现有 Discovery / True OOS Persistent Replay。不要重新下载；先 Run Preflight。",
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("parity_audit", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "parity_audit", running=True, progress=2,
            message="检查 timestamp / timezone / session parity…",
            ok=None, result=None
        )

        result = run_original_parity_audit(
            str(SIP_DATA), str(OOS_DATA),
            str(OUT / "original_engine_parity_audit_v1"),
            progress_cb=cb
        )

        set_job(
            "parity_audit", running=False, progress=100,
            message="✅ Original Engine Parity Audit 完成",
            ok=True, result=result
        )

    except Exception as e:
        set_job(
            "parity_audit", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}", ok=False
        )

@app.post("/start-parity-audit")
def start_parity_audit():
    with LOCK:
        if JOBS["parity_audit"]["running"]:
            return jsonify(JOBS["parity_audit"])
    threading.Thread(target=parity_audit_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["parity_audit"])



def orig_tz_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job(
            "orig_tz", running=False, progress=100,
            message="❌ 缺少现有 Discovery / True OOS Persistent Replay。不要重新下载；先 Run Preflight。",
            ok=False
        )
        return
    try:
        def cb(pct, msg):
            set_job("orig_tz", running=True, progress=int(pct), message=msg, ok=None)

        set_job(
            "orig_tz", running=True, progress=2,
            message="18 parity 已确认；现在做 UTC→NY 输入修复后运行 preserved original engine…",
            ok=None, result=None
        )

        result = run_original_tz_fixed_replay(
            str(SIP_DATA), str(OOS_DATA),
            str(OUT / "original_v01_tzfixed_replay"),
            progress_cb=cb
        )

        set_job(
            "orig_tz", running=False, progress=100,
            message=f"✅ Original TZ-Fixed Replay 完成 · D PF {result['discovery'].get('profit_factor')} / OOS PF {result['true_oos'].get('profit_factor')}",
            ok=True, result=result
        )
    except Exception as e:
        set_job(
            "orig_tz", running=False, progress=100,
            message=f"❌ {type(e).__name__}: {e}", ok=False
        )

@app.post("/start-orig-tz")
def start_orig_tz():
    with LOCK:
        if JOBS["orig_tz"]["running"]:
            return jsonify(JOBS["orig_tz"])
    threading.Thread(target=orig_tz_worker, daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["orig_tz"])



def qc_bench_worker():
    if not SIP_DATA.exists() or not OOS_DATA.exists():
        set_job("qc_bench",running=False,progress=100,message="❌ 缺少现有 Discovery / True OOS Persistent Replay。不要重新下载。",ok=False)
        return
    try:
        def cb(pct,msg): set_job("qc_bench",running=True,progress=int(pct),message=msg,ok=None)
        set_job("qc_bench",running=True,progress=1,message="读取现有两年 Persistent Replay；不下载、不调参数…",ok=None,result=None)
        result=run_qc_stocks_in_play_benchmark(
            str(SIP_DATA),str(OOS_DATA),str(OUT/"qc_stocks_in_play_benchmark_v1"),progress_cb=cb
        )
        set_job("qc_bench",running=False,progress=100,message="✅ QuantConnect External Benchmark 完成",ok=True,result=result)
    except Exception as e:
        set_job("qc_bench",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-qc-bench")
def start_qc_bench():
    with LOCK:
        if JOBS["qc_bench"]["running"]:
            return jsonify(JOBS["qc_bench"])
    threading.Thread(target=qc_bench_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["qc_bench"])



def qc_1m_worker():
    qdir=OUT/"qc_stocks_in_play_benchmark_v1"
    if not (qdir/"qc_oos_strict.csv").exists():
        set_job("qc_1m",running=False,progress=100,message="❌ 先跑 20 QuantConnect External Benchmark；不需要重新下载125股票数据。",ok=False)
        return
    try:
        def cb(pct,msg): set_job("qc_1m",running=True,progress=int(pct),message=msg,ok=None)
        set_job("qc_1m",running=True,progress=1,message="读取20的OOS ambiguity；只下载必要的1min交易日…",ok=None,result=None)
        result=run_qc_1min_oos_resolver(str(qdir),progress_cb=cb)
        set_job("qc_1m",running=False,progress=100,message=f"✅ 1min Resolver 完成 · PF lower {result['lower_bound']['profit_factor']} / upper {result['upper_bound']['profit_factor']}",ok=True,result=result)
    except Exception as e:
        set_job("qc_1m",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-qc-1m")
def start_qc_1m():
    with LOCK:
        if JOBS["qc_1m"]["running"]:
            return jsonify(JOBS["qc_1m"])
    threading.Thread(target=qc_1m_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["qc_1m"])



def bt_parity_worker():
    try:
        def cb(pct,msg): set_job("bt_parity",running=True,progress=int(pct),message=msg,ok=None)
        set_job("bt_parity",running=True,progress=1,message="读取21已缓存的1min ambiguity；不下载、不调参数…",ok=None,result=None)
        result=run_backtrader_execution_parity(progress_cb=cb)
        set_job("bt_parity",running=False,progress=100,
                message=f"✅ Backtrader Parity 完成 · PF {result['backtrader_oos']['profit_factor']} · {'PASS' if result['parity_pass'] else 'FAIL'}",
                ok=True,result=result)
    except Exception as e:
        set_job("bt_parity",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-bt-parity")
def start_bt_parity():
    with LOCK:
        if JOBS["bt_parity"]["running"]:
            return jsonify(JOBS["bt_parity"])
    threading.Thread(target=bt_parity_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["bt_parity"])



def bt_parity_v2_worker():
    try:
        def cb(pct,msg): set_job("bt_parity_v2",running=True,progress=int(pct),message=msg,ok=None)
        set_job("bt_parity_v2",running=True,progress=1,message="真实1min cache Preflight：先测5笔，不通过不长跑…",ok=None,result=None)
        result=run_backtrader_parity_v2(progress_cb=cb)
        set_job("bt_parity_v2",running=False,progress=100,
                message=f"✅ Backtrader v2：{result['verdict']} · resolved {result.get('resolved',0)}/{result.get('ambiguous_trades',161)} · PF {result.get('backtrader_oos',{}).get('profit_factor')}",
                ok=(result["verdict"]=="PASS"),result=result)
    except Exception as e:
        set_job("bt_parity_v2",running=False,progress=100,message=f"❌ {type(e).__name__}: {e}",ok=False)

@app.post("/start-bt-parity-v2")
def start_bt_parity_v2():
    with LOCK:
        if JOBS["bt_parity_v2"]["running"]:
            return jsonify(JOBS["bt_parity_v2"])
    threading.Thread(target=bt_parity_v2_worker,daemon=True).start()
    time.sleep(.05)
    with LOCK:
        return jsonify(JOBS["bt_parity_v2"])


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
