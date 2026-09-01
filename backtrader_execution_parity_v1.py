
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import backtrader as bt
except Exception as e:
    bt = None
    _BT_IMPORT_ERROR = str(e)
else:
    _BT_IMPORT_ERROR = None

QC_DIR = Path("/data/daytrade/results/qc_stocks_in_play_benchmark_v1")
# Fallback path used by earlier app versions.
QC_DIR_ALT = Path("/data/daytrade/qc_stocks_in_play_benchmark_v1")
RESOLUTION_FILE = Path("/data/daytrade/qc_1min_oos_result/qc_oos_1min_resolution.csv")
CACHE_ROOT = Path("/data/daytrade/qc_1min_oos_cache")

def _pf(r):
    r = pd.to_numeric(r, errors="coerce").dropna()
    gp = float(r[r > 0].sum())
    gl = float(-r[r < 0].sum())
    return gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

def _summary_r(r):
    r = pd.Series(pd.to_numeric(r, errors="coerce")).dropna()
    if r.empty:
        return {"trades":0,"win_rate":0.0,"profit_factor":0.0,"avg_r":0.0,"median_r":0.0}
    return {
        "trades": int(len(r)),
        "win_rate": round(float((r>0).mean()),4),
        "profit_factor": round(float(_pf(r)),4),
        "avg_r": round(float(r.mean()),4),
        "median_r": round(float(r.median()),4),
    }

class _OneTradeStrategy(bt.Strategy):
    params = dict(side="LONG", entry=0.0, stop=0.0, activate_time=None)

    def __init__(self):
        self.entry_order = None
        self.stop_order = None
        self.entry_fill = None
        self.exit_fill = None
        self.exit_reason = None
        self.submitted = False

    def next(self):
        dt = self.data.datetime.datetime(0)
        if not self.submitted and dt >= self.p.activate_time - pd.Timedelta(minutes=1):
            self.submitted = True
            if self.p.side == "LONG":
                self.entry_order = self.buy(exectype=bt.Order.Stop, price=float(self.p.entry))
            else:
                self.entry_order = self.sell(exectype=bt.Order.Stop, price=float(self.p.entry))

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        if order is self.entry_order:
            self.entry_fill = float(order.executed.price)
            if self.p.side == "LONG":
                self.stop_order = self.sell(exectype=bt.Order.Stop, price=float(self.p.stop))
            else:
                self.stop_order = self.buy(exectype=bt.Order.Stop, price=float(self.p.stop))
        elif order is self.stop_order:
            self.exit_fill = float(order.executed.price)
            self.exit_reason = "STOP"

    def stop(self):
        if self.entry_fill is not None and self.exit_fill is None:
            self.exit_fill = float(self.data.close[0])
            self.exit_reason = "EOD"

def _run_one_trade(row, bars):
    if bt is None:
        raise RuntimeError(f"backtrader import failed: {_BT_IMPORT_ERROR}")

    entry_time = pd.to_datetime(row["entry_time"], errors="coerce")
    if pd.isna(entry_time):
        raise ValueError("invalid entry_time")

    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars = bars.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
    # One minute before activation is enough to submit the stop-entry so it can trigger on activation minute.
    bars = bars[bars["timestamp"] >= entry_time - pd.Timedelta(minutes=1)].copy()
    if bars.empty:
        raise ValueError("no bars around entry")

    feed_df = bars.set_index("timestamp")[["open","high","low","close","volume"]].copy()
    if "volume" not in feed_df:
        feed_df["volume"] = 0
    feed_df["openinterest"] = 0

    cerebro = bt.Cerebro(stdstats=False, exactbars=1)
    cerebro.broker.setcash(100000.0)
    data = bt.feeds.PandasData(
        dataname=feed_df,
        open="open", high="high", low="low", close="close",
        volume="volume", openinterest="openinterest"
    )
    cerebro.adddata(data)
    cerebro.addstrategy(
        _OneTradeStrategy,
        side=str(row["side"]).upper(),
        entry=float(row["entry"]),
        stop=float(row["stop"]),
        activate_time=entry_time.to_pydatetime()
    )
    st = cerebro.run(runonce=False, preload=False)[0]

    if st.entry_fill is None:
        return {"status":"NO_ENTRY","pnl_r":np.nan}
    risk = abs(float(row["entry"]) - float(row["stop"]))
    if risk <= 0:
        return {"status":"INVALID_RISK","pnl_r":np.nan}
    side = 1.0 if str(row["side"]).upper()=="LONG" else -1.0
    r = (float(st.exit_fill) - float(st.entry_fill)) * side / risk
    return {
        "status":"OK",
        "pnl_r":float(r),
        "entry_fill":float(st.entry_fill),
        "exit_fill":float(st.exit_fill),
        "exit_reason":st.exit_reason
    }

def _locate_qc_dir():
    for p in [QC_DIR, QC_DIR_ALT]:
        if (p/"qc_oos_strict.csv").exists():
            return p
    # Search only the persistent root, no web/no redownload.
    root = Path("/data/daytrade")
    hits = list(root.rglob("qc_oos_strict.csv"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError("qc_oos_strict.csv not found; run 20 first")

def run_backtrader_execution_parity(progress_cb=None):
    if bt is None:
        raise RuntimeError(f"backtrader import failed: {_BT_IMPORT_ERROR}")

    qdir = _locate_qc_dir()
    strict = pd.read_csv(qdir/"qc_oos_strict.csv")
    delayed = pd.read_csv(qdir/"qc_oos_delayed_stop.csv")
    if len(strict) != len(delayed):
        raise ValueError("strict/delayed row count mismatch")

    if not RESOLUTION_FILE.exists():
        raise FileNotFoundError("21 resolver output missing; run 21 first")
    res = pd.read_csv(RESOLUTION_FILE)

    diff = (
        pd.to_numeric(strict["pnl_r"],errors="coerce").round(6) !=
        pd.to_numeric(delayed["pnl_r"],errors="coerce").round(6)
    )
    ambiguous = np.flatnonzero(diff.to_numpy())
    if progress_cb:
        progress_cb(5, f"Backtrader parity: only {len(ambiguous)} ambiguous trades need independent execution")

    # Start from the 21 lower/upper construction.
    lower = pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    upper = pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    for rr in res.to_dict("records"):
        pos = int(rr["row_pos"])
        lo = pd.to_numeric(pd.Series([rr.get("low_r")]),errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([rr.get("high_r")]),errors="coerce").iloc[0]
        if pd.notna(lo): lower.iloc[pos] = float(lo)
        if pd.notna(hi): upper.iloc[pos] = float(hi)
        elif pos < len(delayed): upper.iloc[pos] = float(delayed.iloc[pos]["pnl_r"])

    bt_vec = pd.to_numeric(strict["pnl_r"],errors="coerce").copy()
    rows=[]
    total=max(len(ambiguous),1)

    for n,pos in enumerate(ambiguous,1):
        row = strict.iloc[pos]
        p = CACHE_ROOT/str(row["symbol"])/f"{row['date']}.csv"
        if not p.exists():
            rows.append({"row_pos":int(pos),"symbol":row["symbol"],"date":row["date"],
                         "status":"NO_CACHE","pnl_r":np.nan})
            continue
        bars = pd.read_csv(p)
        try:
            out = _run_one_trade(row,bars)
        except Exception as e:
            out = {"status":"ERROR","pnl_r":np.nan,"error":str(e)}
        rows.append({"row_pos":int(pos),"symbol":row["symbol"],"date":row["date"],**out})
        if np.isfinite(out.get("pnl_r",np.nan)):
            bt_vec.iloc[pos] = float(out["pnl_r"])

        if progress_cb and (n%10==0 or n==total):
            progress_cb(8+int(82*n/total), f"Backtrader event replay {n}/{total}")

    detail = pd.DataFrame(rows)
    outdir = Path("/data/daytrade/backtrader_parity_v1")
    outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outdir/"backtrader_ambiguous_trade_results.csv",index=False)

    lower_s = _summary_r(lower)
    upper_s = _summary_r(upper)
    bt_s = _summary_r(bt_vec)

    ok = detail[detail["status"].eq("OK")] if not detail.empty else detail
    between = 0
    compared = 0
    if not ok.empty:
        for rr in ok.itertuples(index=False):
            pos=int(rr.row_pos)
            lo=float(lower.iloc[pos]); hi=float(upper.iloc[pos]); br=float(rr.pnl_r)
            compared += 1
            if min(lo,hi)-1e-9 <= br <= max(lo,hi)+1e-9:
                between += 1

    result = {
        "study":"Backtrader Independent Execution Parity v1",
        "scope":"True OOS only; strategy selection frozen from 20; 1-minute ambiguity set from 21",
        "engine":"backtrader",
        "backtrader_version":getattr(bt,"__version__","unknown"),
        "oos_trades":int(len(strict)),
        "ambiguous_trades":int(len(ambiguous)),
        "backtrader_resolved":int((detail["status"]=="OK").sum()) if not detail.empty else 0,
        "backtrader_errors":int((detail["status"]!="OK").sum()) if not detail.empty else 0,
        "lower_bound_21":lower_s,
        "upper_bound_21":upper_s,
        "backtrader_oos":bt_s,
        "ambiguous_trade_result_inside_21_bounds_share":round(float(between/compared),4) if compared else 0.0,
        "parity_pass": bool(
            bt_s["profit_factor"] > 1.0 and bt_s["avg_r"] > 0.0 and
            lower_s["profit_factor"] > 1.0 and lower_s["avg_r"] > 0.0 and
            upper_s["profit_factor"] > 1.0 and upper_s["avg_r"] > 0.0
        ),
        "decision_rule":"PASS only if independent Backtrader OOS PF>1 and AvgR>0 while remaining consistent with the positive 21 lower/upper bounds. No strategy parameter tuning."
    }
    (outdir/"backtrader_parity_v1.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    if progress_cb:
        progress_cb(100, f"Backtrader parity complete · PF {bt_s['profit_factor']} · AvgR {bt_s['avg_r']}")
    return result
