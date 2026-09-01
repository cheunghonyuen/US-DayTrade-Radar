
import json
from pathlib import Path
import pandas as pd
import numpy as np

# These are the preserved MVP v0.1 rules. They are NOT tuned here.
CFG = {
  "min_gap_pct": 0.03,
  "max_gap_pct": 0.20,
  "min_rvol": 1.2,
  "opening_range_minutes": 20,
  "entry_start": "10:00",
  "entry_end": "11:30",
}

def _detect_timestamp_col(df):
    for c in ("timestamp","datetime","date"):
        if c in df.columns:
            return c
    raise ValueError("No timestamp/datetime/date column found")

def _normalize_columns(df):
    x = df.copy()
    rename = {}
    aliases = {
        "symbol":["symbol","ticker"],
        "open":["open","o"],
        "high":["high","h"],
        "low":["low","l"],
        "close":["close","c"],
        "volume":["volume","v"],
    }
    for target, names in aliases.items():
        if target in x.columns:
            continue
        for name in names:
            if name in x.columns:
                rename[name] = target
                break
    if rename:
        x = x.rename(columns=rename)
    missing = [c for c in ["symbol","open","high","low","close","volume"] if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return x

def _to_new_york(ts):
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s.dt.tz_convert("America/New_York")

def _build_session_features(df):
    x = _normalize_columns(df)
    tcol = _detect_timestamp_col(x)

    # Current Tiingo-style replay timestamps are treated as UTC, then converted to NY.
    x["timestamp_raw"] = x[tcol]
    x["timestamp_ny"] = _to_new_york(x[tcol])
    x = x.dropna(subset=["timestamp_ny"]).copy()
    x["date_ny"] = x["timestamp_ny"].dt.strftime("%Y-%m-%d")
    x["time_ny"] = x["timestamp_ny"].dt.strftime("%H:%M")

    # Only regular session.
    reg = x[(x["time_ny"] >= "09:30") & (x["time_ny"] < "16:00")].copy()
    reg.sort_values(["symbol","timestamp_ny"], inplace=True)

    # previous regular-session close by symbol/date
    closes = reg.groupby(["symbol","date_ny"], sort=True)["close"].last().reset_index(name="session_close")
    closes["prev_close"] = closes.groupby("symbol")["session_close"].shift(1)
    prev_map = {(str(r.symbol), str(r.date_ny)): float(r.prev_close)
                for r in closes.itertuples(index=False)
                if pd.notna(r.prev_close) and float(r.prev_close) > 0}

    sessions = []
    for (sym, dt), g in reg.groupby(["symbol","date_ny"], sort=True):
        g = g.copy().reset_index(drop=True)
        if g.empty:
            continue

        prev_close = prev_map.get((str(sym), str(dt)))
        open_px = float(g.iloc[0]["open"])
        gap = np.nan if prev_close is None else open_px / prev_close - 1.0

        # 20 minute OR: 09:30 <= t < 09:50
        orb = g[g["time_ny"] < "09:50"].copy()
        or_valid = len(orb) > 0
        orh = float(orb["high"].max()) if or_valid else np.nan
        orl = float(orb["low"].min()) if or_valid else np.nan

        # Candidate entry window
        ent = g[(g["time_ny"] >= CFG["entry_start"]) & (g["time_ny"] <= CFG["entry_end"])].copy()
        breakout_rows = ent[ent["high"] > orh].copy() if or_valid and not ent.empty else ent.iloc[0:0].copy()

        # RVOL compatibility diagnostic:
        # Prefer preserved/current rvol_bar if present. If missing, compute simple bar-volume /
        # symbol minute-of-day historical median within this dataset.
        sessions.append({
            "symbol": str(sym),
            "date": str(dt),
            "rows": int(len(g)),
            "prev_close_ok": prev_close is not None,
            "gap_pct": None if pd.isna(gap) else float(gap),
            "gap_pass": bool(prev_close is not None and gap >= CFG["min_gap_pct"] and gap <= CFG["max_gap_pct"]),
            "or_valid": bool(or_valid),
            "entry_window_rows": int(len(ent)),
            "breakout_count": int(len(breakout_rows)),
            "breakout_pass": bool(len(breakout_rows) > 0),
        })

    return x, reg, pd.DataFrame(sessions)

def _add_rvol_diagnostic(raw_with_ny, reg, sess):
    if sess.empty:
        sess["rvol_pass"] = []
        return sess, {"method":"none","note":"No sessions"}

    reg = reg.copy()
    if "rvol_bar" in reg.columns:
        reg["rvol_diag"] = pd.to_numeric(reg["rvol_bar"], errors="coerce")
        method = "existing_rvol_bar"
    else:
        # simple diagnostic only; not a replacement for original engine logic
        reg["minute"] = reg["time_ny"]
        med = reg.groupby(["symbol","minute"])["volume"].transform("median").replace(0,np.nan)
        reg["rvol_diag"] = pd.to_numeric(reg["volume"], errors="coerce") / med
        method = "diagnostic_volume_over_symbol_minute_median"

    rvol_pass_map = {}
    for (sym, dt), g in reg.groupby(["symbol","date_ny"], sort=True):
        ent = g[(g["time_ny"] >= CFG["entry_start"]) & (g["time_ny"] <= CFG["entry_end"])].copy()
        ok = bool((pd.to_numeric(ent["rvol_diag"], errors="coerce") >= CFG["min_rvol"]).any()) if not ent.empty else False
        rvol_pass_map[(str(sym),str(dt))] = ok

    sess = sess.copy()
    sess["rvol_pass"] = [rvol_pass_map.get((r.symbol,r.date),False) for r in sess.itertuples(index=False)]
    return sess, {
        "method": method,
        "note": "RVOL stage is diagnostic. Exact original engine still remains the authority for final trade generation."
    }

def _funnel(sess):
    total = int(len(sess))
    prev = sess[sess["prev_close_ok"]]
    gap = prev[prev["gap_pass"]]
    orb = gap[gap["or_valid"]]
    entry = orb[orb["entry_window_rows"] > 0]
    breakout = entry[entry["breakout_pass"]]
    rvol = breakout[breakout["rvol_pass"]]
    return {
        "sessions_total": total,
        "prev_close_available": int(len(prev)),
        "gap_3_to_20_pass": int(len(gap)),
        "or20_valid": int(len(orb)),
        "entry_window_has_bars": int(len(entry)),
        "breakout_pass": int(len(breakout)),
        "rvol_diag_pass": int(len(rvol)),
    }

def run_original_parity_audit(discovery_path, oos_path, outdir, progress_cb=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if progress_cb: progress_cb(5, "Parity Audit：Discovery timestamp → America/New_York")
    d_raw = pd.read_csv(discovery_path)
    d_all, d_reg, d_sess = _build_session_features(d_raw)
    d_sess, d_rvol_meta = _add_rvol_diagnostic(d_all, d_reg, d_sess)
    d_fun = _funnel(d_sess)

    if progress_cb: progress_cb(45, "Parity Audit：True OOS timestamp → America/New_York")
    o_raw = pd.read_csv(oos_path)
    o_all, o_reg, o_sess = _build_session_features(o_raw)
    o_sess, o_rvol_meta = _add_rvol_diagnostic(o_all, o_reg, o_sess)
    o_fun = _funnel(o_sess)

    # Time-distribution diagnostics
    def dist(x):
        if x.empty:
            return {}
        s = x["time_ny"].value_counts().sort_index()
        return {
            "min_time": str(x["time_ny"].min()),
            "max_time": str(x["time_ny"].max()),
            "rows_0930_1000": int(((x["time_ny"] >= "09:30") & (x["time_ny"] < "10:00")).sum()),
            "rows_1000_1130": int(((x["time_ny"] >= "10:00") & (x["time_ny"] <= "11:30")).sum()),
            "rows_1550_1600": int(((x["time_ny"] >= "15:50") & (x["time_ny"] < "16:00")).sum()),
        }

    result = {
        "study": "Original Engine Parity Audit v1",
        "timezone_assumption": "Input timestamps parsed as UTC, then converted to America/New_York before session logic.",
        "original_rule_reference": CFG,
        "discovery": {
            "raw_rows": int(len(d_raw)),
            "regular_session_rows_after_tz_fix": int(len(d_reg)),
            "time_distribution": dist(d_reg),
            "funnel": d_fun,
        },
        "true_oos": {
            "raw_rows": int(len(o_raw)),
            "regular_session_rows_after_tz_fix": int(len(o_reg)),
            "time_distribution": dist(o_reg),
            "funnel": o_fun,
        },
        "rvol_diagnostic": {"discovery": d_rvol_meta, "oos": o_rvol_meta},
        "decision_rule": (
            "If regular-session rows and downstream funnel counts become non-zero after timezone normalization, "
            "the previous 0-trade replay was invalid due to input parity/time handling. "
            "Only after this audit should the preserved original engine be rerun on timezone-normalized data."
        )
    }

    d_sess.to_csv(out / "discovery_signal_funnel_sessions.csv", index=False)
    o_sess.to_csv(out / "oos_signal_funnel_sessions.csv", index=False)
    (out / "original_parity_audit_v1.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if progress_cb: progress_cb(100, "Parity Audit 完成")
    return result
