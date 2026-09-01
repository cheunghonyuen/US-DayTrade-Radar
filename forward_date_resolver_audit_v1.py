
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path("/data/daytrade")
PF_ROOT = ROOT / "paper_forward_qc_v1"
STATE = PF_ROOT / "state.json"
LEDGER = PF_ROOT / "paper_ledger.csv"
CACHE5 = PF_ROOT / "cache_5m"
CACHE1 = PF_ROOT / "cache_1m"
OUTDIR = ROOT / "forward_date_resolver_audit_v1"
RESULT = OUTDIR / "result.json"

def _safe_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_read_error": f"{type(e).__name__}: {e}"}

def _dates_from_cache(root):
    out=set()
    if not root.exists():
        return []
    for p in root.rglob("*.csv"):
        s=p.stem
        try:
            d=pd.to_datetime(s, errors="raise").date().isoformat()
            out.add(d)
        except Exception:
            # Some cache layouts may use symbol_date.csv
            parts=s.replace("_","-").split("-")
            if len(parts)>=3:
                cand="-".join(parts[-3:])
                try:
                    out.add(pd.to_datetime(cand, errors="raise").date().isoformat())
                except Exception:
                    pass
    return sorted(out)

def _ledger_dates():
    if not LEDGER.exists():
        return []
    try:
        x=pd.read_csv(LEDGER,usecols=["date"])
        return sorted(set(x["date"].dropna().astype(str)))
    except Exception:
        return []

def _ny_recent_weekdays(now_ny):
    # Pure date resolver diagnostic only; does not claim exchange holiday awareness.
    days=[]
    d=now_ny.date()
    for _ in range(10):
        d=d-timedelta(days=1)
        if d.weekday()<5:
            days.append(d.isoformat())
        if len(days)>=5:
            break
    return days

def run_forward_date_resolver_audit(progress_cb=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if progress_cb: progress_cb(10,"Reading paper-forward state / ledger / cache dates")

    state=_safe_json(STATE)
    processed=sorted(set(str(x) for x in state.get("processed_dates",[])))
    ledger_dates=_ledger_dates()
    cache5_dates=_dates_from_cache(CACHE5)
    cache1_dates=_dates_from_cache(CACHE1)

    now_utc=datetime.now(tz=ZoneInfo("UTC"))
    now_ny=now_utc.astimezone(ZoneInfo("America/New_York"))
    recent_weekdays=_ny_recent_weekdays(now_ny)

    latest_processed=processed[-1] if processed else None
    latest_ledger=ledger_dates[-1] if ledger_dates else None
    latest_cache5=cache5_dates[-1] if cache5_dates else None
    latest_cache1=cache1_dates[-1] if cache1_dates else None

    if progress_cb: progress_cb(50,"Comparing latest completed-looking weekday vs processed/cache state")

    # We only diagnose discovery; do not mutate anything.
    candidate_recent = recent_weekdays[0] if recent_weekdays else None
    reasons=[]

    if candidate_recent:
        if candidate_recent in processed:
            reasons.append("MOST_RECENT_WEEKDAY_ALREADY_MARKED_PROCESSED")
        else:
            reasons.append("MOST_RECENT_WEEKDAY_NOT_IN_PROCESSED_DATES")
        if candidate_recent in cache5_dates:
            reasons.append("MOST_RECENT_WEEKDAY_PRESENT_IN_5M_CACHE")
        else:
            reasons.append("MOST_RECENT_WEEKDAY_MISSING_FROM_5M_CACHE")
        if candidate_recent in ledger_dates:
            reasons.append("MOST_RECENT_WEEKDAY_PRESENT_IN_LEDGER")
        else:
            reasons.append("MOST_RECENT_WEEKDAY_NOT_IN_LEDGER")

    # Strongest likely blocker classification.
    if candidate_recent and candidate_recent not in processed and candidate_recent not in cache5_dates:
        verdict="DATE_DISCOVERY_BLOCKED_BY_MISSING_5M_CACHE"
        next_step="FIX_FORWARD_UPDATE_TO_DISCOVER_AND_DOWNLOAD_NEW_COMPLETED_TRADING_DATE_BEFORE_DECLARING_UP_TO_DATE"
    elif candidate_recent and candidate_recent in processed and candidate_recent not in ledger_dates:
        verdict="STATE_LEDGER_MISMATCH"
        next_step="AUDIT_PROCESSED_DATES_WRITE_PATH_BEFORE_ANY_NEW_FORWARD_APPEND"
    elif candidate_recent and candidate_recent in cache5_dates and candidate_recent not in processed:
        verdict="PENDING_DATE_EXISTS_BUT_UPDATE_SKIPPED"
        next_step="FIX_PENDING_DATE_RESOLVER_ONLY"
    else:
        verdict="NO_SINGLE_BLOCKER_CONFIRMED"
        next_step="INSPECT_FORWARD_UPDATE_DATE_RESOLVER_AND_MARKET_CALENDAR_LOGIC"

    result={
        "study":"Forward Date Resolver Audit v1",
        "verdict":verdict,
        "clock":{
            "utc_now":now_utc.isoformat(),
            "new_york_now":now_ny.isoformat(),
            "recent_ny_weekdays":recent_weekdays
        },
        "state":{
            "processed_dates":processed,
            "latest_processed":latest_processed,
            "date_errors_keys":sorted((state.get("date_errors") or {}).keys())
        },
        "ledger":{
            "dates":ledger_dates,
            "latest_ledger_date":latest_ledger
        },
        "cache_5m":{
            "dates_tail":cache5_dates[-10:],
            "latest_cache_date":latest_cache5,
            "date_count":len(cache5_dates)
        },
        "cache_1m":{
            "dates_tail":cache1_dates[-10:],
            "latest_cache_date":latest_cache1,
            "date_count":len(cache1_dates)
        },
        "diagnostic_flags":reasons,
        "guardrail":"READ_ONLY audit. No strategy parameters, ledger rows, processed_dates, or cache files are changed.",
        "next_step":next_step
    }

    RESULT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if progress_cb:
        progress_cb(100,f"Date Resolver Audit · {verdict}")
    return result
