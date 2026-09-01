
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/data/daytrade")

PATHS = {
    "orb_external": ROOT / "external_challenger_sam_bateman_v1" / "result.json",
    "alpaca_momentum": ROOT / "alpaca_momentum_source_locked_v2" / "result.json",
    "vwap_mean_reversion": ROOT / "external_challenger3_vwap_mean_reversion_v1" / "result.json",
    "pairs_walkforward": ROOT / "pairs_walkforward_v2" / "result.json",
    "paper_ledger": ROOT / "paper_forward_qc_v1" / "paper_ledger.csv",
    "paper_state": ROOT / "paper_forward_qc_v1" / "state.json",
}

def _load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def _paper_metrics():
    lp = PATHS["paper_ledger"]
    sp = PATHS["paper_state"]
    if not lp.exists():
        return {
            "available": False, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_r": 0.0, "median_r": 0.0, "max_dd_r": 0.0, "net_r": 0.0,
            "days": 0
        }
    df = pd.read_csv(lp)
    if df.empty or "pnl_r" not in df.columns:
        return {
            "available": True, "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_r": 0.0, "median_r": 0.0, "max_dd_r": 0.0, "net_r": 0.0,
            "days": 0
        }
    r = pd.to_numeric(df["pnl_r"], errors="coerce").dropna()
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    eq = r.cumsum()
    dd = eq - eq.cummax()
    days = 0
    if sp.exists():
        st = _load_json(sp) or {}
        days = len(set(st.get("processed_dates", [])))
    return {
        "available": True,
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean()), 4) if len(r) else 0.0,
        "profit_factor": round(float(pf), 4),
        "avg_r": round(float(r.mean()), 4) if len(r) else 0.0,
        "median_r": round(float(r.median()), 4) if len(r) else 0.0,
        "max_dd_r": round(float(dd.min()), 4) if len(dd) else 0.0,
        "net_r": round(float(r.sum()), 4),
        "days": int(days),
    }

def _row_orb():
    r = _load_json(PATHS["orb_external"])
    if not r:
        return {"name":"#1 trading-orb","available":False,"status":"MISSING"}
    o = r.get("true_oos", {}).get("metrics", {})
    b = r.get("true_oos", {}).get("bootstrap", {})
    return {
        "name":"#1 trading-orb",
        "family":"ORB",
        "available":True,
        "status":"DROP",
        "sample_type":"Historical True OOS",
        "trades":o.get("trades",0),
        "win_rate":o.get("win_rate",0),
        "profit_factor":o.get("profit_factor",0),
        "avg_edge":o.get("avg_r",0),
        "avg_edge_label":"AvgR",
        "drawdown":o.get("max_dd_r",0),
        "drawdown_label":"MaxDD(R)",
        "bootstrap_pf_gt1":b.get("prob_pf_gt1",0),
        "stability":"Negative OOS",
        "note":"Source-style ORB rules failed on the shared 125-stock pool."
    }

def _row_alpaca():
    r = _load_json(PATHS["alpaca_momentum"])
    if not r:
        return {"name":"#2 Alpaca Momentum","available":False,"status":"MISSING"}
    o = r.get("true_oos", {}).get("metrics", {})
    b = r.get("true_oos", {}).get("bootstrap", {})
    return {
        "name":"#2 Alpaca Momentum",
        "family":"Intraday momentum",
        "available":True,
        "status":"DROP",
        "sample_type":"Historical True OOS",
        "trades":o.get("trades",0),
        "win_rate":o.get("win_rate",0),
        "profit_factor":o.get("profit_factor",0),
        "avg_edge":o.get("avg_r",0),
        "avg_edge_label":"AvgR",
        "drawdown":o.get("max_dd_r",0),
        "drawdown_label":"MaxDD(R)",
        "bootstrap_pf_gt1":b.get("prob_pf_gt1",0),
        "stability":"Near-flat / insufficient evidence",
        "note":"Implementation bug was fixed first; source-locked v2 still failed the pre-declared gate."
    }

def _row_vwap():
    r = _load_json(PATHS["vwap_mean_reversion"])
    if not r:
        return {"name":"#3 VWAP Mean Reversion","available":False,"status":"MISSING"}
    o = r.get("true_oos", {}).get("metrics", {})
    b = r.get("true_oos", {}).get("bootstrap", {})
    return {
        "name":"#3 VWAP Mean Reversion",
        "family":"VWAP mean reversion",
        "available":True,
        "status":"DROP",
        "sample_type":"Historical True OOS",
        "trades":o.get("trades",0),
        "win_rate":o.get("win_rate",0),
        "profit_factor":o.get("profit_factor",0),
        "avg_edge":o.get("avg_trade_pct",0),
        "avg_edge_label":"AvgTrade%",
        "drawdown":o.get("max_dd_pct",0),
        "drawdown_label":"MaxDD%",
        "bootstrap_pf_gt1":b.get("prob_pf_gt1",0),
        "stability":"Strong negative evidence",
        "note":"Very large OOS sample; bootstrap rejected positive PF."
    }

def _row_pairs():
    r = _load_json(PATHS["pairs_walkforward"])
    if not r:
        return {"name":"#4 Pairs Walk-Forward","available":False,"status":"MISSING"}
    o = r.get("overall", {})
    b = r.get("bootstrap", {})
    folds = r.get("fold_count",0)
    pos = r.get("positive_pf_folds",0)
    share = pos/folds if folds else 0.0
    return {
        "name":"#4 Pairs Walk-Forward",
        "family":"Statistical arbitrage",
        "available":True,
        "status":"RESEARCH",
        "sample_type":"18-fold rolling OOS",
        "trades":o.get("trades",0),
        "win_rate":o.get("win_rate",0),
        "profit_factor":o.get("profit_factor",0),
        "avg_edge":o.get("avg_trade_pct",0),
        "avg_edge_label":"AvgTrade%",
        "drawdown":o.get("max_dd_pct",0),
        "drawdown_label":"MaxDD%",
        "bootstrap_pf_gt1":b.get("prob_pf_gt1",0),
        "stability":f"{pos}/{folds} folds PF>1 ({share:.1%})",
        "note":"Closest to flat, but PF/Sharpe/bootstrap/fold-stability gate not passed."
    }

def run_champion_benchmark_summary():
    p = _paper_metrics()
    champion = {
        "name":"26 Frozen Forward Champion",
        "family":"Stocks-in-Play ORB",
        "available":p["available"],
        "status":"FORWARD_TEST",
        "sample_type":"True future Paper Forward",
        "trades":p["trades"],
        "days":p["days"],
        "win_rate":p["win_rate"],
        "profit_factor":p["profit_factor"],
        "avg_edge":p["avg_r"],
        "avg_edge_label":"AvgR",
        "drawdown":p["max_dd_r"],
        "drawdown_label":"MaxDD(R)",
        "net_r":p["net_r"],
        "bootstrap_pf_gt1":None,
        "stability":f"{p['days']} completed forward day(s)",
        "note":"Only strategy currently in true future forward test. Do not judge live-readiness before 20 days AND 100 trades."
    }

    challengers = [_row_orb(), _row_alpaca(), _row_vwap(), _row_pairs()]
    available = [x for x in challengers if x.get("available")]

    result = {
        "study":"Champion vs External Benchmarks Summary v1",
        "champion":champion,
        "challengers":challengers,
        "summary":{
            "external_tested":len(available),
            "external_promoted":sum(1 for x in available if x.get("status")=="PROMOTE"),
            "external_dropped":sum(1 for x in available if x.get("status")=="DROP"),
            "external_research_only":sum(1 for x in available if x.get("status")=="RESEARCH"),
            "champion_is_live_ready":bool(p["days"]>=20 and p["trades"]>=100),
        },
        "interpretation":[
            "External strategies were not all identical in timeframe or metric units; compare PF, evidence quality, and gate outcome more than raw drawdown units.",
            "The four external candidates did not beat the pre-declared evidence gates on our data.",
            "The Frozen Forward Champion is not declared successful yet; its current advantage is that it remains the only candidate still alive in true future data.",
            "No external failure should be used to tune the Frozen Champion."
        ],
        "next_gate":"Continue Frozen Forward unchanged. Re-open external research only for candidates that pass an evidence screen before coding."
    }
    return result
