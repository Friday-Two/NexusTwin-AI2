import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from digital_twin import negotiate, save_trace
from execution_verifier import execute_agreement, verify_persisted_booking


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def main():
    profile_path = ROOT / "profiles" / "digital_twin_profile.json"
    request_path = ROOT / "profiles" / "sample_request.txt"

    evidence_dir = ROOT / "experiments" / "phase7_end_to_end"
    negotiation_dir = ROOT / "experiments" / "digital_twin_negotiation"
    db_path = ROOT / "verification" / "services.db"

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    request = request_path.read_text(encoding="utf-8").strip()

    run_id = "phase7-demo-001"
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "steps": {},
        "overall_status": "failed"
    }

    # Step 1: Negotiate.
    trace = negotiate(request, profile, max_rounds=3)
    save_trace(trace, negotiation_dir / "phase7_negotiation_trace.json")
    write_json(evidence_dir / "negotiation_trace.json", trace)

    summary["steps"]["negotiation"] = {
        "status": trace.get("status"),
        "rounds": len(trace.get("rounds", [])),
        "privacy_audit": trace.get("privacy_audit", {})
    }

    print("=== STEP 1: NEGOTIATION ===")
    print("Status:", trace.get("status"))
    print("Rounds:", len(trace.get("rounds", [])))
    print("Privacy audit:", trace.get("privacy_audit", {}))

    if trace.get("status") != "agreement" or not trace.get("accepted_offer"):
        summary["failure_reason"] = trace.get(
            "failure_reason", "No valid agreement was produced"
        )
        write_json(evidence_dir / "run_summary.json", summary)
        write_json(evidence_dir / "failure_trace.json", trace)
        print("\nSTOP: No agreement; execution was not attempted.")
        print("Failure evidence saved:", evidence_dir)
        return 1

    agreement = trace["accepted_offer"]
    print("Accepted offer:", json.dumps(agreement, indent=2))

    # Step 2: Execute the accepted agreement in SQLite.
    execution = execute_agreement(
        agreement=agreement,
        profile=profile,
        db_path=db_path,
        booking_id=run_id
    )
    write_json(evidence_dir / "execution_record.json", execution)

    summary["steps"]["execution"] = execution
    print("\n=== STEP 2: SIMULATED EXECUTION ===")
    print(json.dumps(execution, indent=2))

    if execution.get("status") != "persisted":
        summary["failure_reason"] = "Execution did not persist the agreement"
        write_json(evidence_dir / "run_summary.json", summary)
        print("\nSTOP: Execution failed.")
        return 2

    # Step 3: Independently verify the persisted booking.
    verification = verify_persisted_booking(
        booking_id=run_id,
        expected_agreement=agreement,
        profile=profile,
        db_path=db_path
    )
    write_json(evidence_dir / "verification_report.json", verification)

    summary["steps"]["verification"] = verification
    print("\n=== STEP 3: INDEPENDENT VERIFICATION ===")
    print(json.dumps(verification, indent=2))

    if verification.get("verified") is not True:
        summary["failure_reason"] = "Independent verification failed"
        write_json(evidence_dir / "run_summary.json", summary)
        print("\nFAIL: Booking was persisted but did not pass verification.")
        return 3

    # Step 4: Check the privacy audit and record final outcome.
    privacy_audit = trace.get("privacy_audit", {})
    privacy_ok = (
        privacy_audit.get("phone_sharing_allowed") is False
        and privacy_audit.get("phone_disclosed_to_agents") is False
        and privacy_audit.get("private_fields_in_outgoing_offers") is False
    )

    summary["steps"]["privacy_policy"] = {
        "passed": privacy_ok,
        "audit": privacy_audit
    }

    if not privacy_ok:
        summary["failure_reason"] = "Privacy audit did not pass"
        write_json(evidence_dir / "run_summary.json", summary)
        print("\nFAIL: Privacy audit did not pass.")
        return 4

    summary["overall_status"] = "passed"
    summary["steps"]["final_agreement_validation"] = trace.get(
        "final_agreement_validation"
    )
    write_json(evidence_dir / "run_summary.json", summary)

    print("\n=== END-TO-END RESULT ===")
    print("PASS: Negotiation")
    print("PASS: SQLite persistence")
    print("PASS: Independent verification")
    print("PASS: Privacy audit")
    print("PASS: Run summary and evidence saved")
    print("Evidence directory:", evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
