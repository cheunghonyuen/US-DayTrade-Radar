import json
from pathlib import Path

def run_opening_drive_decision(tournament_json, audit_json, outdir):
    tp, ap = Path(tournament_json), Path(audit_json)
    if not tp.exists():
        raise FileNotFoundError(f"missing Tournament result: {tp}")
    if not ap.exists():
        raise FileNotFoundError(f"missing Opening Drive Audit result: {ap}")

    tournament = json.loads(tp.read_text(encoding="utf-8"))
    audit = json.loads(ap.read_text(encoding="utf-8"))

    ranked = tournament.get("results") or []
    alternatives = [x for x in ranked if x.get("setup") != "OPENING_DRIVE"]
    next_candidate = alternatives[0] if alternatives else None

    audit_verdict = audit.get("verdict")
    passed = int(audit.get("checks_passed", 0))
    total = int(audit.get("checks_total", 0))
    failed_checks = [k for k, v in (audit.get("checks") or {}).items() if not bool(v)]

    # Predeclared decision rule. No parameter fitting, no replay, no redownload.
    if audit_verdict == "AUDIT_PASS":
        action = "KEEP_OPENING_DRIVE"
        mainline = "OPENING_DRIVE"
        reason = "Opening Drive passed every frozen audit check."
    else:
        action = "DROP_OPENING_DRIVE_FROM_MAINLINE"
        mainline = next_candidate.get("setup") if next_candidate else None
        reason = (
            f"Opening Drive did not pass the frozen audit ({passed}/{total}); "
            "do not repair it on the same samples. Promote the next Tournament-ranked setup."
        )

    result = {
        "study": "Opening Drive Decision v1",
        "audit_verdict": audit_verdict,
        "audit_checks": f"{passed}/{total}",
        "failed_checks": failed_checks,
        "decision": action,
        "next_mainline": mainline,
        "next_candidate": next_candidate,
        "rule": "AUDIT_PASS => KEEP. Anything else => DROP from mainline; no same-sample repair.",
        "reason": reason,
        "guardrails": [
            "No threshold tuning.",
            "No historical replay.",
            "No market-data redownload.",
            "Do not rescue Opening Drive on Discovery/OOS already used for selection/audit."
        ]
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "opening_drive_decision_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
