
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from paper_forward_test_v1 import run_paper_forward_update
from paper_command_center_v1 import load_command_center
from champion_benchmark_summary_v1 import run_champion_benchmark_summary

ROOT = Path("/data/daytrade")
PF_ROOT = ROOT / "paper_forward_qc_v1"
LEDGER = PF_ROOT / "paper_ledger.csv"
STATE = PF_ROOT / "state.json"
FREEZE = PF_ROOT / "freeze_manifest.json"
CACHE5 = PF_ROOT / "cache_5m"
CACHE1 = PF_ROOT / "cache_1m"
OPS_ROOT = ROOT / "forward_daily_ops_v1"
OPS_STATE = OPS_ROOT / "ops_state.json"

GATE_DAYS = 20
GATE_TRADES = 100

def _safe_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _cache_count(root, date_str):
    if not root.exists():
        return 0
    n = 0
    for p in root.glob(f"*/{date_str}.csv"):
        if p.exists() and p.stat().st_size > 20:
            n += 1
    return n

def _date_completeness(date_str, state):
    e = (state.get("date_errors") or {}).get(date_str, [])
    c5 = _cache_count(CACHE5, date_str)
    c1 = _cache_count(CACHE1, date_str)
    # 5-minute data should exist broadly for the 125-symbol universe.
    # 1-minute files only exist for selected candidates, so no 125-file requirement.
    return {
        "date": date_str,
        "cache_5m_symbols": int(c5),
        "cache_1m_symbols": int(c1),
        "logged_errors": int(len(e)),
        "complete_enough": bool(c5 >= 100 and len(e) <= 25),
        "note": "5m >=100/125 and <=25 logged symbol errors" if c5 >= 100 and len(e) <= 25
                else "REVIEW: sparse 5m cache or elevated download errors"
    }

def _ledger_daily():
    if not LEDGER.exists():
        return []
    x = pd.read_csv(LEDGER)
    if x.empty or "date" not in x.columns or "pnl_r" not in x.columns:
        return []
    x["date"] = x["date"].astype(str)
    x["pnl_r"] = pd.to_numeric(x["pnl_r"], errors="coerce")
    x = x.dropna(subset=["pnl_r"])
    out = []
    for d,g in x.groupby("date", sort=True):
        r = g["pnl_r"]
        gp = float(r[r>0].sum()); gl = float(-r[r<0].sum())
        pf = gp/gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        out.append({
            "date": str(d),
            "trades": int(len(g)),
            "win_rate": round(float((r>0).mean()),4),
            "pf": round(float(pf),4),
            "net_r": round(float(r.sum()),4)
        })
    return out

def _final_gate(command):
    g = command.get("gate", {})
    days = int(g.get("days",0))
    trades = int(g.get("trades",0))
    ready = bool(days >= GATE_DAYS and trades >= GATE_TRADES)
    return {
        "days": days,
        "days_target": GATE_DAYS,
        "days_remaining": max(0, GATE_DAYS-days),
        "trades": trades,
        "trades_target": GATE_TRADES,
        "trades_remaining": max(0, GATE_TRADES-trades),
        "ready_for_final_audit": ready,
        "status": "FINAL_AUDIT_READY" if ready else "ACCUMULATING_FUTURE_DATA"
    }

def run_forward_daily_ops(progress_cb=None):
    OPS_ROOT.mkdir(parents=True, exist_ok=True)

    if not FREEZE.exists():
        raise FileNotFoundError("26 freeze manifest missing. Frozen strategy must exist before Daily Ops.")

    if progress_cb:
        progress_cb(5, "Step 1/4 · Frozen manifest verified; append-only update")

    update = run_paper_forward_update(
        progress_cb=(lambda p,m: progress_cb(min(55, 8 + int((p or 0)*0.47)), f"26 update · {m}"))
        if progress_cb else None
    )

    if progress_cb:
        progress_cb(60, "Step 2/4 · Read-only Command Center / Gate")

    command = load_command_center()
    state = _safe_json(STATE)
    processed = [str(x) for x in state.get("processed_dates", [])]
    newly = [str(x) for x in update.get("new_dates_processed", [])]

    dates_to_check = newly if newly else (processed[-1:] if processed else [])
    completeness = [_date_completeness(d, state) for d in dates_to_check]

    if progress_cb:
        progress_cb(75, "Step 3/4 · Refresh Champion vs External Benchmarks")

    benchmark = run_champion_benchmark_summary()
    gate = _final_gate(command)
    daily = _ledger_daily()

    latest = daily[-1] if daily else None
    data_ok = all(x["complete_enough"] for x in completeness) if completeness else True

    result = {
        "study":"Forward Daily Ops & Gate Monitor v1",
        "timestamp_utc":datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "mode":"FROZEN_APPEND_ONLY",
        "update_verdict":update.get("verdict","UPDATED"),
        "new_dates_processed":newly,
        "new_trades":int(update.get("new_trades",0)),
        "data_completeness":completeness,
        "data_integrity_ok":bool(data_ok),
        "overall":command.get("overall",{}),
        "latest_day":latest,
        "gate":gate,
        "benchmark_summary":benchmark.get("summary",{}),
        "processed_dates":processed,
        "guardrails":[
            "Frozen strategy parameters are never changed.",
            "Previously processed dates are reused; old ledger rows are not recalculated.",
            "External #1-#4 benchmarks are read-only and never rerun by Daily Ops.",
            "Gate judgement is forbidden before BOTH 20 completed forward days and 100 trades."
        ],
        "next_action":"RUN_FINAL_FROZEN_FORWARD_AUDIT" if gate["ready_for_final_audit"]
                      else "WAIT_FOR_NEXT_COMPLETED_US_TRADING_DAY"
    }

    OPS_STATE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress_cb:
        o=result["overall"]
        progress_cb(
            100,
            f"Daily Ops complete · {gate['days']}/{GATE_DAYS} days · "
            f"{gate['trades']}/{GATE_TRADES} trades · PF {o.get('profit_factor',0)} · "
            f"{gate['status']}"
        )
    return result
