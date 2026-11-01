import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# Resolve the project root independently of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "digital_twin" / "src"
sys.path.insert(0, str(SRC_DIR))

from ai import NegotiationInference


OUTPUT_PATH = Path(__file__).resolve().parent / "inference_result.json"


def main():
    own_input = "1 4 4 1 1 2"
    partner_input = "1 4 4 1 1 2"

    dialogue = (
        "You: I can offer one unit of item0 and one unit of item2. "
        "Partner: I would prefer more item1. "
        "You: Let us try to reach a mutually acceptable allocation."
    )

    report = {
        "experiment": "Phase 9 - FLAN-T5 Inference",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "corrected_fp32",
        "device": "cpu",
        "inputs": {
            "own_input": own_input,
            "partner_input": partner_input,
            "dialogue": dialogue,
        },
        "status": "failed",
        "model_load_seconds": None,
        "prediction": None,
        "execution_performed": False,
    }

    try:
        print("Loading corrected FLAN-T5 checkpoint...")

        load_start = time.perf_counter()

        engine = NegotiationInference(
            checkpoint_name="corrected_fp32",
            device="cpu",
            max_input_length=512,
            max_new_tokens=128,
            num_beams=1,
        )

        report["model_load_seconds"] = (
            time.perf_counter() - load_start
        )

        print("Model loaded successfully.")
        print("Running inference...")

        result = engine.predict(
            own_input=own_input,
            partner_input=partner_input,
            dialogue=dialogue,
        )

        report["prediction"] = {
            "raw_output": result.raw_output,
            "category": result.category,
            "is_supported": result.is_supported,
            "allocation_valid": result.allocation_valid,
            "validation_reason": result.validation_reason,
            "is_acceptable": result.is_acceptable,
            "latency_seconds": result.latency_seconds,
            "device": result.device,
        }

        report["status"] = "completed"

        print("\n========== PHASE 9 INFERENCE ==========")
        print("Output:", result.raw_output)
        print("Category:", result.category)
        print("Supported:", result.is_supported)
        print("Allocation valid:", result.allocation_valid)
        print("Validation:", result.validation_reason)
        print("Acceptable:", result.is_acceptable)
        print("Model load:", f"{report['model_load_seconds']:.4f}s")
        print("Inference:", f"{result.latency_seconds:.4f}s")
        print("Execution performed: False")
        print("=======================================")

    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise

    finally:
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\nExperiment report saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()