import json
from pathlib import Path
from digital_twin import negotiate, save_trace

ROOT = Path(__file__).resolve().parents[1]

def main():
    profile_path = ROOT / "profiles" / "digital_twin_profile.json"
    request_path = ROOT / "profiles" / "sample_request.txt"
    trace_path = (
        ROOT / "experiments" / "digital_twin_negotiation"
        / "negotiation_trace.json"
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    request = request_path.read_text(encoding="utf-8").strip()

    trace = negotiate(request, profile, max_rounds=3)
    save_trace(trace, trace_path)

    print("=== DIGITAL TWIN NEGOTIATION ===")
    print("Status:", trace["status"])
    print("Rounds executed:", len(trace["rounds"]))
    print("Privacy audit:", trace["privacy_audit"])

    if trace["accepted_offer"]:
        print("Accepted offer:", json.dumps(
            trace["accepted_offer"], indent=2
        ))
    else:
        print("Failure:", trace["failure_reason"])

    print("Trace saved:", trace_path)

if __name__ == "__main__":
    main()
