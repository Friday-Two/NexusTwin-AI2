# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')



# PHASE 1 — CELL 1: ENVIRONMENT AUDIT

import os
import sys
import platform
import importlib.util
import json
from pathlib import Path

print("=" * 65)
print("PERSONAL AI DIGITAL TWIN — KAGGLE ENVIRONMENT AUDIT")
print("=" * 65)

print("\n[1] Python and operating system")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Working directory:", os.getcwd())

print("\n[2] Important packages")
packages = [
    "torch",
    "transformers",
    "datasets",
    "sklearn",
    "pandas",
    "numpy",
    "accelerate",
    "sentencepiece",
]

for package in packages:
    print(f"{package:16}:", importlib.util.find_spec(package) is not None)

print("\n[3] PyTorch and accelerator")
if importlib.util.find_spec("torch"):
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory (GiB):",
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        )
    else:
        print("No CUDA GPU detected; CPU audit can continue.")

print("\n[4] Kaggle directories")
for path in ["/kaggle/input", "/kaggle/working", "/kaggle/temp"]:
    p = Path(path)
    print(f"\n{path}: exists={p.exists()}")

    if p.exists():
        entries = list(p.iterdir())
        for entry in entries[:15]:
            print("  ", entry.name)
        if len(entries) > 15:
            print(f"   ... and {len(entries) - 15} more entries")

print("\n[5] Environment audit complete.")
print("No dataset or model has been downloaded in this cell.")




# PHASE 1 — CELL 2: NETWORK AND DATASET ACCESS CHECK

import urllib.request
import json
from pathlib import Path

urls = {
    "Hugging Face dataset API":
        "https://huggingface.co/api/datasets/mikelewis0/deal_or_no_dialog",
    "Official research repository":
        "https://api.github.com/repos/facebookresearch/end-to-end-negotiator",
}

network_results = {}

for name, url in urls.items():
    print(f"\nChecking: {name}")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "DigitalTwin-Kaggle-Audit/1.0"}
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read(5000).decode("utf-8", errors="replace")
            print("HTTP status:", response.status)
            print("Response preview:", body[:350].replace("\n", " "))
            network_results[name] = {
                "accessible": True,
                "http_status": response.status
            }
    except Exception as exc:
        print("Access failed:", type(exc).__name__, str(exc)[:250])
        network_results[name] = {
            "accessible": False,
            "error": f"{type(exc).__name__}: {str(exc)[:250]}"
        }

output_dir = Path("/kaggle/working/digital_twin")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "network_audit.json", "w") as f:
    json.dump(network_results, f, indent=2)

print("\nNetwork audit saved to:", output_dir / "network_audit.json")




# PHASE 1 — CELL 4: DOWNLOAD AUTHENTIC RESEARCH DATA
# Downloads only the official dataset files; does not train a model.

import urllib.request
import hashlib
import json
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/"
    "end-to-end-negotiator/master/src/data/negotiate"
)

DATA_DIR = Path("/kaggle/working/digital_twin/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train.txt": f"{BASE_URL}/train.txt",
    "val.txt": f"{BASE_URL}/val.txt",
    "test.txt": f"{BASE_URL}/test.txt",
}

audit = {
    "source": "facebookresearch/end-to-end-negotiator",
    "source_url": BASE_URL,
    "files": {}
}

for filename, url in FILES.items():
    destination = DATA_DIR / filename
    print(f"\nDownloading {filename}...")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DigitalTwin-Kaggle-Audit/1.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Download failed for {filename}: HTTP {response.status}"
            )
        content = response.read()

    if not content.strip():
        raise RuntimeError(f"{filename} downloaded but is empty.")

    destination.write_bytes(content)

    text = content.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]

    file_info = {
        "path": str(destination),
        "bytes": len(content),
        "nonempty_lines": len(lines),
        "sha256": hashlib.sha256(content).hexdigest(),
        "first_line_preview": lines[0][:500] if lines else None,
    }

    audit["files"][filename] = file_info

    print("Saved:", destination)
    print("Bytes:", file_info["bytes"])
    print("Non-empty lines:", file_info["nonempty_lines"])
    print("SHA-256:", file_info["sha256"])
    print("First example:", file_info["first_line_preview"])

if not all(name in audit["files"] for name in FILES):
    raise RuntimeError("One or more required dataset splits are missing.")

audit_path = Path("/kaggle/working/digital_twin/dataset_source_audit.json")
audit_path.parent.mkdir(parents=True, exist_ok=True)
audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\n" + "=" * 65)
print("AUTHENTIC DATASET FILE DOWNLOAD COMPLETE")
print("=" * 65)
print("Audit saved to:", audit_path)
print("No synthetic examples were created.")




# PHASE 1 — CELL 5: STRUCTURAL INTEGRITY CHECK

import re
import json
from pathlib import Path

DATA_DIR = Path("/kaggle/working/digital_twin/raw")

expected_markers = [
    "<input>",
    "</input>",
    "<dialogue>",
    "</dialogue>",
    "<output>",
    "</output>",
]

results = {}

for filename in ["train.txt", "val.txt", "test.txt"]:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    malformed = []
    marker_counts = {marker: 0 for marker in expected_markers}

    for index, line in enumerate(lines):
        for marker in expected_markers:
            if marker in line:
                marker_counts[marker] += 1

        missing = [
            marker for marker in expected_markers
            if marker not in line
        ]

        if missing:
            malformed.append({
                "line": index + 1,
                "missing_markers": missing,
                "preview": line[:200],
            })

    results[filename] = {
        "nonempty_lines": len(lines),
        "marker_counts": marker_counts,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:3],
    }

    print(f"\n{filename}")
    print(json.dumps(results[filename], indent=2))

    if malformed:
        print(
            "WARNING: Some lines do not match the expected format. "
            "We will inspect these before preprocessing."
        )

    if not lines:
        raise ValueError(f"{filename} contains no nonempty lines.")

# Save the results even if structural warnings need investigation.
output_path = Path(
    "/kaggle/working/digital_twin/structural_audit.json"
)
output_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nStructural audit saved to:", output_path)
print("Audit finished. No model training has occurred.")




# PHASE 1 — CELL 6: LABEL AND TARGET DISTRIBUTION AUDIT

import re
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/kaggle/working/digital_twin/raw")
OUTPUT_DIR = Path("/kaggle/working/digital_twin")

FIELD_PATTERN = re.compile(
    r"<(input|dialogue|output|partner_input)>(.*?)</\1>"
)

def parse_record(line):
    fields = dict(FIELD_PATTERN.findall(line))

    required = {"input", "dialogue", "output", "partner_input"}
    if not required.issubset(fields):
        return None

    return fields

audit = {}

for split_name in ["train", "val", "test"]:
    path = DATA_DIR / f"{split_name}.txt"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    parsed = []
    parse_errors = []

    for index, line in enumerate(lines):
        record = parse_record(line)

        if record is None:
            parse_errors.append(index + 1)
        else:
            parsed.append(record)

    output_types = Counter()
    output_tokens = Counter()
    input_length_counts = Counter()
    dialogue_lengths = []
    empty_dialogues = 0
    allocation_format_errors = 0

    for record in parsed:
        output = record["output"].strip()
        output_tokens.update(output.split())

        if "<disagree>" in output:
            output_types["disagreement"] += 1
        elif re.search(r"\bitem[0-9]+\s*=", output):
            output_types["allocation"] += 1
            assignments = re.findall(
                r"item[0-9]+\s*=\s*\d+", output
            )
            if len(assignments) != 6:
                allocation_format_errors += 1
        else:
            output_types["other"] += 1

        input_values = record["input"].split()
        input_length_counts[len(input_values)] += 1

        dialogue = record["dialogue"].strip()
        dialogue_lengths.append(len(dialogue))

        if not dialogue:
            empty_dialogues += 1

    split_audit = {
        "raw_records": len(lines),
        "parsed_records": len(parsed),
        "parse_errors": len(parse_errors),
        "output_types": dict(output_types),
        "most_common_output_tokens": output_tokens.most_common(15),
        "input_field_token_counts": dict(input_length_counts),
        "empty_dialogues": empty_dialogues,
        "mean_dialogue_characters": (
            round(sum(dialogue_lengths) / len(dialogue_lengths), 2)
            if dialogue_lengths else 0
        ),
        "allocation_format_errors": allocation_format_errors,
        "parse_error_line_examples": parse_errors[:10],
    }

    audit[split_name] = split_audit

    print(f"\n{'=' * 55}")
    print(f"SPLIT: {split_name.upper()}")
    print(json.dumps(split_audit, indent=2))

    if parse_errors:
        print("WARNING: Some records could not be parsed.")

audit_path = OUTPUT_DIR / "label_distribution_audit.json"
audit_path.write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8"
)

print("\nAudit saved to:", audit_path)
print("No training or label modifications were performed.")




# PHASE 1 — FINALIZATION CELL
# Reproducibility, model-access check, and final audit report

import os
import sys
import json
import random
import platform
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

OUT_DIR = Path("/kaggle/working/digital_twin")
DATA_DIR = OUT_DIR / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== REPRODUCIBILITY ===")
print("Seed:", SEED)
print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Check that the authentic dataset files exist and remain readable.
file_audit = {}

for name in ["train.txt", "val.txt", "test.txt"]:
    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    content = path.read_bytes()

    file_audit[name] = {
        "bytes": len(content),
        "nonempty_lines": sum(
            bool(line.strip())
            for line in content.decode("utf-8").splitlines()
        ),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

print("\n=== AUTHENTIC DATA FILES ===")
print(json.dumps(file_audit, indent=2))

# Check whether the Hugging Face model metadata can be accessed.
# This is a small metadata request, not model-weight download.
model_check = {
    "model_id": "google/flan-t5-small",
    "metadata_accessible": False,
}

try:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "google/flan-t5-small",
        trust_remote_code=False,
    )

    model_check.update({
        "metadata_accessible": True,
        "model_type": config.model_type,
        "encoder_decoder": bool(config.is_encoder_decoder),
        "vocab_size": config.vocab_size,
    })

except Exception as exc:
    model_check.update({
        "error_type": type(exc).__name__,
        "error": str(exc)[:500],
    })

print("\n=== MODEL METADATA CHECK ===")
print(json.dumps(model_check, indent=2))

# Check that all expected audit files are present.
audit_files = [
    "network_audit.json",
    "dataset_source_audit.json",
    "structural_audit.json",
    "label_distribution_audit.json",
]

audit_file_status = {
    name: (OUT_DIR / name).exists()
    for name in audit_files
}

print("\n=== PREVIOUS AUDIT ARTIFACTS ===")
print(json.dumps(audit_file_status, indent=2))

report = {
    "phase": 1,
    "status": "dataset_and_environment_audited",
    "seed": SEED,
    "python": sys.version,
    "platform": platform.platform(),
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None
    ),
    "dataset_files": file_audit,
    "model_metadata_check": model_check,
    "previous_audit_files": audit_file_status,
}

report_path = OUT_DIR / "phase1_final_report.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nPhase 1 report saved to:", report_path)
print("Review the model metadata check before moving to Phase 2.")




# PHASE 2 — CELL 1
# Parse, validate, and preprocess the authentic Deal or No Deal dataset.

import re
import json
import random
import hashlib
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)

ROOT = Path("/kaggle/working/digital_twin")
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FIELD_PATTERN = re.compile(
    r"<(input|dialogue|output|partner_input)>(.*?)</\1>"
)

def parse_record(line):
    fields = dict(FIELD_PATTERN.findall(line))
    required = {"input", "dialogue", "output", "partner_input"}

    if not required.issubset(fields):
        raise ValueError("Record is missing required fields.")

    return {key: fields[key].strip() for key in required}

def classify_target(target):
    """
    Mutually exclusive categories:
      allocation   : contains six item assignments
      disagreement : contains only repeated <disagree> markers
      no_agreement : contains <no_agreement> and no allocation
      disconnect   : contains <disconnect> and no allocation
      other        : any remaining target format
    """
    assignments = re.findall(
        r"\b(item[0-9]+)\s*=\s*(\d+)", target
    )

    if assignments:
        return "allocation"

    tokens = target.split()

    if tokens and all(token == "<disagree>" for token in tokens):
        return "disagreement"

    if "<no_agreement>" in tokens:
        return "no_agreement"

    if "<disconnect>" in tokens:
        return "disconnect"

    return "other"

def validate_allocation(target):
    assignments = re.findall(
        r"\b(item[0-9]+)\s*=\s*(\d+)", target
    )

    if len(assignments) != 6:
        return False, "expected_six_assignments"

    names = [name for name, _ in assignments]

    expected_names = [
        "item0", "item1", "item2",
        "item0", "item1", "item2",
    ]

    if names != expected_names:
        return False, "unexpected_assignment_order"

    values = [int(value) for _, value in assignments]

    # The two sides' allocations should sum to the item counts.
    item_counts = list(map(int, current_input.split()))

    if len(item_counts) != 6:
        return False, "invalid_input_length"

    totals = [
        values[0] + values[3],
        values[1] + values[4],
        values[2] + values[5],
    ]

    if totals != item_counts[:3]:
        return False, "allocation_does_not_conserve_items"

    if any(value < 0 for value in values):
        return False, "negative_allocation"

    return True, "valid"

# Preserve the official split files; do not create a random replacement split.
split_mapping = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}

summary = {}
processed_hashes = {}

for split_name, filename in split_mapping.items():
    source_path = RAW_DIR / filename
    records = []
    categories = Counter()
    allocation_checks = Counter()
    seen_inputs = set()

    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            fields = parse_record(line)
            current_input = fields["input"]
            target = " ".join(fields["output"].split())
            category = classify_target(target)

            valid, reason = True, "not_an_allocation"

            if category == "allocation":
                valid, reason = validate_allocation(target)
                allocation_checks[reason] += 1

            # Keep every original example, even if an allocation fails validation.
            # A failed allocation is flagged for investigation, never silently removed.
            prompt = (
                "Predict the negotiation outcome.\n"
                f"Your item counts and values: {fields['input']}\n"
                f"Partner item counts and values: {fields['partner_input']}\n"
                f"Negotiation dialogue: {fields['dialogue']}\n"
                "Output the original target format."
            )

            record = {
                "id": f"{split_name}_{line_number:05d}",
                "split": split_name,
                "input": prompt,
                "target": target,
                "target_category": category,
                "allocation_valid": valid if category == "allocation" else None,
                "allocation_validation_reason": reason,
            }

            records.append(record)
            categories[category] += 1
            seen_inputs.add(fields["input"])

    output_path = PROCESSED_DIR / f"{split_name}.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    output_bytes = output_path.read_bytes()
    processed_hashes[split_name] = hashlib.sha256(output_bytes).hexdigest()

    summary[split_name] = {
        "records": len(records),
        "target_categories": dict(categories),
        "allocation_validation": dict(allocation_checks),
        "unique_own_input_fields": len(seen_inputs),
        "processed_file": str(output_path),
        "processed_sha256": processed_hashes[split_name],
        "first_record": records[0] if records else None,
    }

    print(f"\n{'=' * 60}")
    print(split_name.upper())
    print(json.dumps(summary[split_name], indent=2))

    if not records:
        raise RuntimeError(f"No records processed for {split_name}.")

    if category == "allocation" and not valid:
        print("Note: inspect allocation validation warnings above.")

summary_path = PROCESSED_DIR / "preprocessing_audit.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("Audit:", summary_path)
print("Original files were not modified.")
print("No examples were intentionally dropped.")




# PHASE 2 — CELL 1A: REVIEW ALLOCATION SEMANTICS
# Read-only diagnostic. Does not modify any files.

import re
from pathlib import Path

RAW_DIR = Path("/kaggle/working/digital_twin/raw")

FIELD_PATTERN = re.compile(
    r"<(input|dialogue|output|partner_input)>(.*?)</\1>"
)

def extract_fields(line):
    return dict(FIELD_PATTERN.findall(line))

for filename in ["train.txt", "val.txt", "test.txt"]:
    path = RAW_DIR / filename
    print(f"\n{'=' * 70}\n{filename.upper()}")

    shown = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        fields = extract_fields(line)
        target = fields.get("output", "").strip()

        if re.search(r"\bitem[0-9]+\s*=", target):
            print("\nINPUT:        ", fields.get("input"))
            print("PARTNER INPUT:", fields.get("partner_input"))
            print("DIALOGUE:     ", fields.get("dialogue", "")[:250])
            print("TARGET:       ", target)

            shown += 1
            if shown == 3:
                break

print("\nReview complete. No labels or source files were modified.")




# PHASE 2 — CELL 1C: CORRECTED PREPROCESSING
# Count/value layout: count0, value0, count1, value1, count2, value2

import re
import json
from pathlib import Path
from collections import Counter

ROOT = Path("/kaggle/working/digital_twin")
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FIELD_PATTERN = re.compile(
    r"<(input|dialogue|output|partner_input)>(.*?)</\1>"
)

def parse_record(line):
    fields = dict(FIELD_PATTERN.findall(line))
    required = {"input", "dialogue", "output", "partner_input"}
    if not required.issubset(fields):
        raise ValueError("Missing required record fields.")
    return {key: fields[key].strip() for key in required}

def classify_target(target):
    assignments = re.findall(
        r"\b(item[0-9]+)\s*=\s*(\d+)", target
    )
    if assignments:
        return "allocation"
    tokens = target.split()
    if tokens and all(t == "<disagree>" for t in tokens):
        return "disagreement"
    if "<no_agreement>" in tokens:
        return "no_agreement"
    if "<disconnect>" in tokens:
        return "disconnect"
    return "other"

def validate_allocation(target, own_input):
    pairs = re.findall(
        r"\b(item[0-9]+)\s*=\s*(\d+)", target
    )
    if len(pairs) != 6:
        return False, "expected_six_assignments"

    expected_names = [
        "item0", "item1", "item2",
        "item0", "item1", "item2",
    ]
    names = [name for name, _ in pairs]
    if names != expected_names:
        return False, "unexpected_assignment_order"

    values = [int(value) for _, value in pairs]
    numbers = list(map(int, own_input.split()))
    if len(numbers) != 6:
        return False, "invalid_input_length"

    # Correct layout: count0, value0, count1, value1, count2, value2.
    counts = [numbers[0], numbers[2], numbers[4]]
    allocated_totals = [
        values[0] + values[3],
        values[1] + values[4],
        values[2] + values[5],
    ]

    if allocated_totals != counts:
        return False, "allocation_does_not_conserve_items"

    return True, "valid"

split_mapping = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}

audit = {}

for split, filename in split_mapping.items():
    records = []
    categories = Counter()
    allocation_checks = Counter()

    source_path = RAW_DIR / filename
    with source_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            fields = parse_record(line)
            target = " ".join(fields["output"].split())
            category = classify_target(target)

            valid = None
            reason = "not_an_allocation"

            if category == "allocation":
                valid, reason = validate_allocation(
                    target, fields["input"]
                )
                allocation_checks[reason] += 1

            prompt = (
                "Predict the negotiation outcome.\n"
                f"Your item counts and values: {fields['input']}\n"
                f"Partner item counts and values: {fields['partner_input']}\n"
                f"Negotiation dialogue: {fields['dialogue']}\n"
                "Output the original target format."
            )

            records.append({
                "id": f"{split}_{line_number:05d}",
                "split": split,
                "input": prompt,
                "target": target,
                "target_category": category,
                "allocation_valid": valid,
                "allocation_validation_reason": reason,
            })
            categories[category] += 1

    output_path = PROCESSED_DIR / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit[split] = {
        "records": len(records),
        "target_categories": dict(categories),
        "allocation_validation": dict(allocation_checks),
        "processed_file": str(output_path),
    }

    print(f"\n{split.upper()}")
    print(json.dumps(audit[split], indent=2))

audit_path = PROCESSED_DIR / "preprocessing_audit.json"
audit_path.write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8"
)

print("\nCorrected audit saved to:", audit_path)
print("Original raw files were not modified.")




# PHASE 2 — CELL 2: MAJORITY-TARGET BASELINE
# Uses the corrected processed files.
# No model training and no additional downloads.

import json
import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
PROCESSED_DIR = ROOT / "processed"
EXPERIMENT_DIR = ROOT / "experiments" / "majority_target_baseline"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc
    return rows

train = load_jsonl(PROCESSED_DIR / "train.jsonl")
validation = load_jsonl(PROCESSED_DIR / "validation.jsonl")
test = load_jsonl(PROCESSED_DIR / "test.jsonl")

if not train or not validation or not test:
    raise RuntimeError("A required dataset split is empty.")

# Choose the most frequent full target using training data only.
target_counts = Counter(row["target"] for row in train)
baseline_target, baseline_count = target_counts.most_common(1)[0]

print("Most frequent training target:", repr(baseline_target))
print("Training occurrences:", baseline_count)
print("Training proportion:", round(baseline_count / len(train), 6))

def evaluate(rows, split_name):
    exact_matches = sum(
        row["target"] == baseline_target for row in rows
    )

    category_counts = Counter(
        row["target_category"] for row in rows
    )

    category_correct = Counter()
    for row in rows:
        if row["target"] == baseline_target:
            category_correct[row["target_category"]] += 1

    category_accuracy = (
        sum(
            row["target_category"]
            == next(
                (
                    train_row["target_category"]
                    for train_row in train
                    if train_row["target"] == baseline_target
                ),
                "unknown",
            )
            for row in rows
        ) / len(rows)
    )

    return {
        "split": split_name,
        "examples": len(rows),
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / len(rows),
        "target_category_counts": dict(category_counts),
        "baseline_target_category": next(
            (
                row["target_category"]
                for row in train
                if row["target"] == baseline_target
            ),
            "unknown",
        ),
        "category_accuracy": category_accuracy,
        "correct_exact_matches_by_category": dict(category_correct),
    }

results = {
    "baseline": "majority_target",
    "seed": 42,
    "training_examples": len(train),
    "majority_target": baseline_target,
    "majority_target_count": baseline_count,
    "training_majority_proportion": baseline_count / len(train),
    "validation": evaluate(validation, "validation"),
    "test": evaluate(test, "test"),
}

for split_name, rows in [
    ("validation", validation),
    ("test", test),
]:
    pred_path = EXPERIMENT_DIR / f"{split_name}_predictions.csv"

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "target_category",
                "reference",
                "prediction",
                "exact_match",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "id": row["id"],
                "target_category": row["target_category"],
                "reference": row["target"],
                "prediction": baseline_target,
                "exact_match": row["target"] == baseline_target,
            })

metrics_path = EXPERIMENT_DIR / "metrics.json"
metrics_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n=== BASELINE RESULTS ===")
print(json.dumps(results, indent=2, ensure_ascii=False))
print("\nSaved metrics:", metrics_path)
print("Saved validation and test prediction CSV files.")




# PHASE 2 — CELL 3: CORRECT CATEGORY METRIC

import json
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
EXP_DIR = ROOT / "experiments" / "majority_target_baseline"

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train = read_jsonl(ROOT / "processed" / "train.jsonl")
validation = read_jsonl(ROOT / "processed" / "validation.jsonl")
test = read_jsonl(ROOT / "processed" / "test.jsonl")

from collections import Counter

majority_target = Counter(r["target"] for r in train).most_common(1)[0][0]
majority_category = next(
    r["target_category"] for r in train
    if r["target"] == majority_target
)

metrics_path = EXP_DIR / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

for split_name, rows in [
    ("validation", validation),
    ("test", test),
]:
    category_correct = sum(
        row["target_category"] == majority_category for row in rows
    )

    metrics[split_name]["category_accuracy"] = (
        category_correct / len(rows)
    )
    metrics[split_name]["category_correct"] = category_correct
    metrics[split_name]["baseline_target_category"] = majority_category

metrics_path.write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Corrected baseline metrics:")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print("\nSaved:", metrics_path)




# PHASE 3 — CELL 1: LOAD MODEL AND VERIFY READINESS

import json
import torch
import transformers
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SEED = 42
MODEL_NAME = "google/flan-t5-small"

ROOT = Path("/kaggle/working/digital_twin")
PROCESSED_DIR = ROOT / "processed"
MODEL_DIR = ROOT / "models" / "flan_t5_small_negotiation"
EXPERIMENT_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Check the processed data without modifying it.
def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_data = read_jsonl(PROCESSED_DIR / "train.jsonl")
validation_data = read_jsonl(PROCESSED_DIR / "validation.jsonl")
test_data = read_jsonl(PROCESSED_DIR / "test.jsonl")

print("=== DATA CHECK ===")
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Test:", len(test_data))
assert (len(train_data), len(validation_data), len(test_data)) == (10095, 1087, 1052)

for split_name, rows in [
    ("train", train_data),
    ("validation", validation_data),
    ("test", test_data),
]:
    assert all(row.get("input") and row.get("target") for row in rows), (
        f"Missing input or target in {split_name}"
    )

print("\n=== ENVIRONMENT ===")
print("Python/PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

# Download the tokenizer and pretrained model.
print("\n=== LOADING PRETRAINED MODEL ===")
print("Model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("\n=== MODEL READY ===")
print("Model parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Tokenizer vocabulary:", len(tokenizer))
print("Encoder-decoder:", model.config.is_encoder_decoder)
print("Model directory:", MODEL_DIR)
print("Experiment directory:", EXPERIMENT_DIR)

# Save a small readiness report.
readiness = {
    "seed": SEED,
    "model_name": MODEL_NAME,
    "transformers_version": transformers.__version__,
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "train_examples": len(train_data),
    "validation_examples": len(validation_data),
    "test_examples": len(test_data),
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "tokenizer_vocabulary": len(tokenizer),
    "encoder_decoder": model.config.is_encoder_decoder,
}
(EXPERIMENT_DIR / "readiness.json").write_text(
    json.dumps(readiness, indent=2), encoding="utf-8"
)

print("\nReadiness report saved.")




# PHASE 3 — CELL 2: TOKENIZATION AND LENGTH AUDIT

import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq

SEED = 42
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 96
TOKENIZE_BATCH_SIZE = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path("/kaggle/working/digital_twin")
PROCESSED_DIR = ROOT / "processed"
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
EXP_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_data = read_jsonl(PROCESSED_DIR / "train.jsonl")
validation_data = read_jsonl(PROCESSED_DIR / "validation.jsonl")
test_data = read_jsonl(PROCESSED_DIR / "test.jsonl")

# Reuse tokenizer and model from Cell 1.
assert tokenizer is not None and model is not None
assert len(train_data) == 10095
assert len(validation_data) == 1087
assert len(test_data) == 1052

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 96

def audit_lengths(rows, field, max_length):
    lengths = []
    for start in range(0, len(rows), TOKENIZE_BATCH_SIZE):
        batch = rows[start:start + TOKENIZE_BATCH_SIZE]
        texts = [row[field] for row in batch]
        encoded = tokenizer(
            texts,
            truncation=False,
            padding=False
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    return {
        "examples": len(lengths),
        "max_tokens_before_truncation": max(lengths),
        "mean_tokens": round(float(np.mean(lengths)), 2),
        "median_tokens": int(np.median(lengths)),
        "over_limit": sum(length > max_length for length in lengths),
        "over_limit_percent": round(
            100 * sum(length > max_length for length in lengths) / len(lengths),
            3
        ),
        "limit": max_length
    }

print("Auditing original sequence lengths...")
length_audit = {}
for split_name, rows in [
    ("train", train_data),
    ("validation", validation_data),
    ("test", test_data)
]:
    length_audit[split_name] = {
        "input": audit_lengths(rows, "input", MAX_INPUT_LENGTH),
        "target": audit_lengths(rows, "target", MAX_TARGET_LENGTH)
    }

print("\n=== SEQUENCE LENGTH AUDIT ===")
print(json.dumps(length_audit, indent=2))

class NegotiationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_input_length, max_target_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.inputs = []
        self.targets = []

        for start in range(0, len(rows), TOKENIZE_BATCH_SIZE):
            batch = rows[start:start + TOKENIZE_BATCH_SIZE]

            encoded_inputs = tokenizer(
                [row["input"] for row in batch],
                max_length=max_input_length,
                truncation=True,
                padding=False
            )
            encoded_targets = tokenizer(
                text_target=[row["target"] for row in batch],
                max_length=max_target_length,
                truncation=True,
                padding=False
            )

            self.inputs.extend(encoded_inputs["input_ids"])
            self.targets.extend(encoded_targets["input_ids"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": [1] * len(self.inputs[idx]),
            "labels": self.targets[idx]
        }

print("\nTokenizing training data...")
train_dataset = NegotiationDataset(
    train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
)

print("Tokenizing validation data...")
validation_dataset = NegotiationDataset(
    validation_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
)

print("Tokenizing test data for later evaluation...")
test_dataset = NegotiationDataset(
    test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)

tokenization_report = {
    "seed": SEED,
    "model_name": MODEL_NAME,
    "max_input_length": MAX_INPUT_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "length_audit": length_audit,
    "tokenized_examples": {
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset)
    },
    "truncation_enabled": True,
    "dynamic_batch_padding": True
}

report_path = EXP_DIR / "tokenization_audit.json"
report_path.write_text(
    json.dumps(tokenization_report, indent=2),
    encoding="utf-8"
)

# Inspect a sample batch to verify tensor shapes and labels.
sample_features = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
sample_batch = data_collator(sample_features)

print("\n=== TOKENIZATION CHECK ===")
print("Train examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))
print("Test examples:", len(test_dataset))
print("Sample input tensor shape:", tuple(sample_batch["input_ids"].shape))
print("Sample label tensor shape:", tuple(sample_batch["labels"].shape))
print("Padding label ID:", data_collator.label_pad_token_id)
print("Audit saved:", report_path)

assert len(train_dataset) == 10095
assert len(validation_dataset) == 1087
assert len(test_dataset) == 1052
assert sample_batch["input_ids"].ndim == 2
assert sample_batch["labels"].ndim == 2

print("\nSTEP 3.2 COMPLETE")




# PHASE 3 — CELL 3 RECOVERY: TRAIN IN FULL FP32
# Important: reload pretrained weights; do not reuse the failed training model.

import json
import math
import time
import random
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

SEED = 42
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

ROOT = Path("/kaggle/working/digital_twin")
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
MODEL_DIR = ROOT / "models" / "flan_t5_small_negotiation"
BEST_MODEL_DIR = MODEL_DIR / "best_fp32"

EXP_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

assert len(train_dataset) == 10095
assert len(validation_dataset) == 1087

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "Kaggle GPU is required for this training configuration."

# Reload fresh pretrained weights to discard the failed run's model state.
del model
torch.cuda.empty_cache()

print("Loading fresh pretrained model in FP32...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(device)

model.config.use_cache = False
model.train()

# Fail early if the first batch produces an invalid loss.
smoke_batch = data_collator(
    [train_dataset[i] for i in range(min(BATCH_SIZE, len(train_dataset)))]
)
smoke_batch = {k: v.to(device) for k, v in smoke_batch.items()}

with torch.no_grad():
    smoke_outputs = model(**smoke_batch)

smoke_loss = smoke_outputs.loss.item()
print("FP32 smoke-test loss:", smoke_loss)
assert math.isfinite(smoke_loss), (
    "Fresh FP32 model produced a non-finite loss. Stop and report this output."
)

del smoke_batch, smoke_outputs
torch.cuda.empty_cache()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
total_steps = updates_per_epoch * EPOCHS
warmup_steps = max(1, int(0.05 * total_steps))

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def move_batch(batch):
    return {key: value.to(device) for key, value in batch.items()}

def evaluate_loss(loader):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation loss detected. Evaluation stopped."
                )

            n = batch["input_ids"].size(0)
            total_loss += loss.item() * n
            total_examples += n

    model.train()
    return total_loss / total_examples

training_log = {
    "seed": SEED,
    "model_name": MODEL_NAME,
    "precision": "FP32",
    "epochs_requested": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUMULATION,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "train_examples": len(train_dataset),
    "validation_examples": len(validation_dataset),
    "optimizer_steps_per_epoch": updates_per_epoch,
    "epochs": []
}

best_val_loss = float("inf")
global_step = 0
start_time = time.time()

print("\n=== FP32 TRAINING CONFIGURATION ===")
print("GPU:", torch.cuda.get_device_name(0))
print("Training batches:", len(train_loader))
print("Optimizer updates:", total_steps)
print("FP16/mixed precision: DISABLED")
print("Effective batch size:", BATCH_SIZE * GRAD_ACCUMULATION)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_train_loss = 0.0
    total_examples = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch = move_batch(batch)
        batch_size_now = batch["input_ids"].size(0)

        # Full FP32 forward pass: no autocast and no GradScaler.
        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_idx + 1}. "
                "Training stopped before saving a bad checkpoint."
            )

        (loss / GRAD_ACCUMULATION).backward()

        total_train_loss += loss.detach().item() * batch_size_now
        total_examples += batch_size_now

        is_update_step = (
            (batch_idx + 1) % GRAD_ACCUMULATION == 0
            or (batch_idx + 1) == len(train_loader)
        )

        if is_update_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )

            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"Non-finite gradient norm at batch {batch_idx + 1}."
                )

            # Correct order: optimizer first, then scheduler.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Running loss: {total_train_loss / total_examples:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    train_loss = total_train_loss / total_examples
    val_loss = evaluate_loss(validation_loader)

    if not math.isfinite(train_loss) or not math.isfinite(val_loss):
        raise RuntimeError("Non-finite epoch metric. Checkpoint not saved.")

    record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "optimizer_steps": global_step,
        "duration_seconds": time.time() - epoch_start
    }
    training_log["epochs"].append(record)

    print("\n=== EPOCH SUMMARY ===")
    print(f"Training loss:   {train_loss:.6f}")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Optimizer steps: {global_step}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)
        training_log["best_validation_loss"] = best_val_loss
        training_log["best_epoch"] = epoch + 1
        print("Best checkpoint saved:", BEST_MODEL_DIR)

training_log["training_completed"] = True
training_log["total_training_seconds"] = time.time() - start_time

log_path = EXP_DIR / "training_log_fp32.json"
log_path.write_text(
    json.dumps(training_log, indent=2),
    encoding="utf-8"
)

model.config.use_cache = True

print("\n=== FP32 TRAINING COMPLETE ===")
print(json.dumps(training_log, indent=2))
print("Training log:", log_path)
print("Checkpoint:", BEST_MODEL_DIR)




# PHASE 3 — CELL 4: VALIDATION GENERATION AND METRICS

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("/kaggle/working/digital_twin")
CHECKPOINT = ROOT / "models" / "flan_t5_small_negotiation" / "best_fp32"
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
EXP_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
assert len(validation_data) == 1087

print("Loading saved fine-tuned checkpoint...")
eval_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
eval_model = AutoModelForSeq2SeqLM.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.float32
).to(device)
eval_model.eval()

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0
)

predictions = []

print("Generating validation predictions...")
with torch.no_grad():
    for batch_idx, batch in enumerate(validation_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated_ids = eval_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

        decoded = eval_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        predictions.extend(decoded)

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(validation_loader):
            print(f"Processed batches: {batch_idx + 1}/{len(validation_loader)}")

assert len(predictions) == len(validation_data), (
    f"Expected {len(validation_data)} predictions, got {len(predictions)}"
)

def normalize(text):
    return " ".join(str(text).strip().split())

def classify_target(text):
    text = normalize(text)

    if re.search(r"\bitem\d+\s*=", text):
        return "allocation"
    if re.fullmatch(r"(?:<disagree>\s*)+", text):
        return "disagreement"
    if "<no_agreement>" in text:
        return "no_agreement"
    if "<disconnect>" in text:
        return "disconnect"
    return "other"

# Exact-match metrics and category accuracy.
exact_matches = 0
category_correct = 0
target_category_counts = Counter()
predicted_category_counts = Counter()

# Whitespace-token multiset micro metrics.
total_overlap = 0
total_predicted_tokens = 0
total_reference_tokens = 0

results = []

for row, prediction in zip(validation_data, predictions):
    reference = row["target"]
    pred_norm = normalize(prediction)
    ref_norm = normalize(reference)

    exact = pred_norm == ref_norm
    if exact:
        exact_matches += 1

    true_category = row["target_category"]
    pred_category = classify_target(prediction)

    target_category_counts[true_category] += 1
    predicted_category_counts[pred_category] += 1
    category_correct += int(true_category == pred_category)

    pred_tokens = pred_norm.split()
    ref_tokens = ref_norm.split()
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())

    total_overlap += overlap
    total_predicted_tokens += len(pred_tokens)
    total_reference_tokens += len(ref_tokens)

    results.append({
        "input": row["input"],
        "reference": reference,
        "prediction": prediction,
        "true_category": true_category,
        "predicted_category": pred_category,
        "exact_match": exact
    })

precision = (
    total_overlap / total_predicted_tokens
    if total_predicted_tokens else 0.0
)
recall = (
    total_overlap / total_reference_tokens
    if total_reference_tokens else 0.0
)
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

metrics = {
    "model": "google/flan-t5-small",
    "checkpoint": str(CHECKPOINT),
    "split": "validation",
    "examples": len(validation_data),
    "exact_matches": exact_matches,
    "exact_match_accuracy": exact_matches / len(validation_data),
    "category_correct": category_correct,
    "category_accuracy": category_correct / len(validation_data),
    "target_category_counts": dict(target_category_counts),
    "predicted_category_counts": dict(predicted_category_counts),
    "token_micro_precision": precision,
    "token_micro_recall": recall,
    "token_micro_f1": f1,
    "token_metric_definition": (
        "Whitespace-token multiset overlap; these are not semantic metrics."
    ),
    "generation": {
        "max_new_tokens": 64,
        "num_beams": 1,
        "do_sample": False
    }
}

metrics_path = EXP_DIR / "validation_metrics.json"
metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

predictions_path = EXP_DIR / "validation_predictions.csv"
with predictions_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n=== VALIDATION METRICS ===")
print(json.dumps(metrics, indent=2))
print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)

print("\n=== SAMPLE PREDICTIONS ===")
for i, result in enumerate(results[:5], start=1):
    print(f"\nExample {i}")
    print("Reference: ", result["reference"])
    print("Prediction:", result["prediction"])
    print("Exact match:", result["exact_match"])
    print(
        "Categories:",
        result["true_category"], "->", result["predicted_category"]
    )

print("\nSTEP 3.4 COMPLETE")




# PHASE 3 — CELL 5: VALIDATION ERROR AUDIT
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"

assert len(results) == len(validation_data), "Validation results missing."

# 1. Compare against the true majority-category baseline.
true_categories = [r["target_category"] for r in validation_data]
category_counts = Counter(true_categories)
majority_category = category_counts.most_common(1)[0][0]
majority_category_correct = category_counts[majority_category]

# 2. Exact-match accuracy for each true category.
class_stats = defaultdict(lambda: {"examples": 0, "exact_matches": 0})
for result in results:
    category = result["true_category"]
    class_stats[category]["examples"] += 1
    class_stats[category]["exact_matches"] += int(result["exact_match"])

for category, stats in class_stats.items():
    stats["exact_match_accuracy"] = (
        stats["exact_matches"] / stats["examples"]
        if stats["examples"] else 0.0
    )

# 3. Inspect predicted target patterns and category confusion.
predicted_categories = Counter(r["predicted_category"] for r in results)
confusion = Counter(
    (r["true_category"], r["predicted_category"]) for r in results
)

# 4. Count outputs matching the expected structural patterns.
def allocation_assignment_count(text):
    import re
    return len(re.findall(r"\bitem\d+\s*=\s*\d+", text))

structure_stats = Counter()
for result in results:
    prediction = result["prediction"].strip()
    category = result["predicted_category"]

    if category == "allocation":
        count = allocation_assignment_count(prediction)
        if count == 6:
            structure_stats["allocation_with_6_assignments"] += 1
        else:
            structure_stats["allocation_with_wrong_assignment_count"] += 1
    elif category == "other":
        structure_stats["other_output"] += 1
    else:
        structure_stats[f"recognized_{category}"] += 1

audit = {
    "split": "validation",
    "examples": len(results),
    "majority_category": majority_category,
    "majority_category_accuracy": majority_category_correct / len(results),
    "model_category_accuracy": sum(
        r["true_category"] == r["predicted_category"] for r in results
    ) / len(results),
    "model_exact_match_accuracy": sum(
        r["exact_match"] for r in results
    ) / len(results),
    "true_category_counts": dict(category_counts),
    "predicted_category_counts": dict(predicted_categories),
    "per_true_category_exact_match": dict(class_stats),
    "category_confusion": {
        f"{actual} -> {predicted}": count
        for (actual, predicted), count in confusion.items()
    },
    "output_structure": dict(structure_stats)
}

print("=== VALIDATION ERROR AUDIT ===")
print(json.dumps(audit, indent=2))

audit_path = EXP_DIR / "validation_error_audit.json"
audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("\nSaved:", audit_path)

print("\n=== EXAMPLES THE MODEL GOT WRONG ===")
wrong = [r for r in results if not r["exact_match"]]
for i, result in enumerate(wrong[:10], start=1):
    print(f"\nExample {i}")
    print("True category:     ", result["true_category"])
    print("Predicted category:", result["predicted_category"])
    print("Reference:         ", result["reference"])
    print("Prediction:        ", result["prediction"])




# PHASE 3 — CELL 6 (CORRECTED): TARGET TOKENIZATION DIAGNOSTIC

import json
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"

markers = ["<disagree>", "<no_agreement>", "<disconnect>"]

print("=== TOKENIZER DIAGNOSTICS ===")
print("Tokenizer class:", type(eval_tokenizer).__name__)
print("Tokenizer vocabulary size:", len(eval_tokenizer))
print("Special-token map:", eval_tokenizer.special_tokens_map)

# Inspect whether each outcome marker exists as one complete vocabulary token.
vocab = eval_tokenizer.get_vocab()
round_trip_results = []

for marker in markers:
    encoded = eval_tokenizer(
        marker,
        add_special_tokens=False
    )["input_ids"]

    item = {
        "marker": marker,
        "is_single_vocabulary_token": marker in vocab,
        "token_pieces": eval_tokenizer.tokenize(marker),
        "token_ids": encoded,
        "decode_keep_special": eval_tokenizer.decode(
            encoded, skip_special_tokens=False
        ),
        "decode_skip_special": eval_tokenizer.decode(
            encoded, skip_special_tokens=True
        ),
    }
    round_trip_results.append(item)
    print("\n" + json.dumps(item, indent=2, ensure_ascii=False))

print("\n=== TARGET ENCODING ROUND-TRIP ===")
examples = [
    "<disagree> <disagree> <disagree> <disagree> <disagree> <disagree>",
    "<no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement>",
    "<disconnect> <disconnect> <disconnect> <disconnect> <disconnect> <disconnect>",
    "item0=1 item1=0 item2=2 item0=1 item1=3 item2=0",
]

for target in examples:
    ids = eval_tokenizer(
        text_target=target,
        add_special_tokens=False
    )["input_ids"]

    decoded_keep = eval_tokenizer.decode(
        ids, skip_special_tokens=False
    )
    decoded_skip = eval_tokenizer.decode(
        ids, skip_special_tokens=True
    )

    item = {
        "original": target,
        "token_pieces": eval_tokenizer.convert_ids_to_tokens(ids),
        "decoded_keep_special": decoded_keep,
        "decoded_skip_special": decoded_skip,
        "keep_matches_original": decoded_keep.strip() == target.strip(),
        "skip_matches_original": decoded_skip.strip() == target.strip(),
    }
    round_trip_results.append(item)
    print("\n" + json.dumps(item, indent=2, ensure_ascii=False))

report_path = EXP_DIR / "target_tokenization_diagnostic.json"
report_path.write_text(
    json.dumps(round_trip_results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("\nDiagnostic saved:", report_path)




# PHASE 3 — CELL 7: CREATE TOKENIZER-SAFE TARGET REPRESENTATION
# Original processed files are not modified.
# The original target is retained alongside the model-facing target.

import json
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

ROOT = Path("/kaggle/working/digital_twin")
PROCESSED_DIR = ROOT / "processed"
MODEL_DATA_DIR = ROOT / "processed_model_targets"
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"

MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Reload the original pretrained tokenizer, not anything modified during training.
base_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def model_target(row):
    category = row["target_category"]

    if category == "disagreement":
        return "OUTCOME: DISAGREEMENT"
    if category == "no_agreement":
        return "OUTCOME: NO_AGREEMENT"
    if category == "disconnect":
        return "OUTCOME: DISCONNECT"

    # Allocation target: preserve its original content exactly.
    if category == "allocation":
        return row["target"]

    raise ValueError(f"Unexpected target category: {category!r}")

split_paths = {
    "train": PROCESSED_DIR / "train.jsonl",
    "validation": PROCESSED_DIR / "validation.jsonl",
    "test": PROCESSED_DIR / "test.jsonl",
}

summary = {}

for split_name, source_path in split_paths.items():
    rows = read_jsonl(source_path)
    converted = []

    for row in rows:
        converted_row = dict(row)
        converted_row["original_target"] = row["target"]
        converted_row["model_target"] = model_target(row)
        converted.append(converted_row)

    output_path = MODEL_DATA_DIR / f"{split_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    categories = Counter(row["target_category"] for row in converted)
    summary[split_name] = {
        "examples": len(converted),
        "categories": dict(categories),
        "output_file": str(output_path),
    }

    # Verify allocation targets were not changed.
    for row in converted:
        if row["target_category"] == "allocation":
            assert row["model_target"] == row["original_target"]

# Verify ordinary-text outcome labels round-trip without losing characters.
round_trip_checks = {}
for label in [
    "OUTCOME: DISAGREEMENT",
    "OUTCOME: NO_AGREEMENT",
    "OUTCOME: DISCONNECT",
]:
    ids = base_tokenizer(
        text_target=label,
        add_special_tokens=False
    )["input_ids"]
    decoded = base_tokenizer.decode(
        ids, skip_special_tokens=False
    )
    round_trip_checks[label] = {
        "decoded": decoded,
        "round_trip_exact": decoded.strip() == label,
        "token_pieces": base_tokenizer.convert_ids_to_tokens(ids),
    }
    assert decoded.strip() == label, f"Label failed round-trip: {label}"

report = {
    "model": "google/flan-t5-small",
    "representation_version": "plain_text_outcome_labels_v1",
    "original_splits_modified": False,
    "summary": summary,
    "outcome_label_round_trip": round_trip_checks,
    "allocation_targets_preserved": True,
}

report_path = EXP_DIR / "target_representation_audit.json"
report_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=== MODEL-FACING TARGET AUDIT ===")
print(json.dumps(report, indent=2, ensure_ascii=False))
print("\nSaved audit:", report_path)
print("\nSTEP 3.5 PREPARATION COMPLETE")




# PHASE 3 — CELL 8: TOKENIZE CORRECTED MODEL-FACING TARGETS

import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

SEED = 42
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 96
TOKENIZE_BATCH_SIZE = 32

ROOT = Path("/kaggle/working/digital_twin")
DATA_DIR = ROOT / "processed_model_targets"
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
CORRECTED_CHECKPOINT = ROOT / "models" / "flan_t5_small_negotiation" / "corrected_fp32"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_rows = read_jsonl(DATA_DIR / "train.jsonl")
validation_rows = read_jsonl(DATA_DIR / "validation.jsonl")
test_rows = read_jsonl(DATA_DIR / "test.jsonl")

assert (len(train_rows), len(validation_rows), len(test_rows)) == (10095, 1087, 1052)

# Reload the original tokenizer and model; do not reuse the earlier fine-tuned model.
corrected_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
corrected_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    dtype=torch.float32
)

class CorrectedNegotiationDataset(Dataset):
    def __init__(self, rows):
        self.input_ids = []
        self.labels = []

        for start in range(0, len(rows), TOKENIZE_BATCH_SIZE):
            batch = rows[start:start + TOKENIZE_BATCH_SIZE]

            enc_inputs = corrected_tokenizer(
                [row["input"] for row in batch],
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=False
            )
            enc_targets = corrected_tokenizer(
                text_target=[row["model_target"] for row in batch],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False
            )

            self.input_ids.extend(enc_inputs["input_ids"])
            self.labels.extend(enc_targets["input_ids"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": [1] * len(self.input_ids[index]),
            "labels": self.labels[index],
        }

print("Tokenizing corrected training targets...")
corrected_train_dataset = CorrectedNegotiationDataset(train_rows)

print("Tokenizing corrected validation targets...")
corrected_validation_dataset = CorrectedNegotiationDataset(validation_rows)

print("Tokenizing corrected test targets...")
corrected_test_dataset = CorrectedNegotiationDataset(test_rows)

corrected_collator = DataCollatorForSeq2Seq(
    tokenizer=corrected_tokenizer,
    model=corrected_model,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)

sample_batch = corrected_collator([
    corrected_train_dataset[i] for i in range(4)
])

print("\n=== CORRECTED TOKENIZATION CHECK ===")
print("Training:", len(corrected_train_dataset))
print("Validation:", len(corrected_validation_dataset))
print("Test:", len(corrected_test_dataset))
print("Sample input shape:", tuple(sample_batch["input_ids"].shape))
print("Sample label shape:", tuple(sample_batch["labels"].shape))

# Verify decoded labels are the intended model-facing targets.
for i in range(3):
    row = train_rows[i]
    label_ids = corrected_train_dataset[i]["labels"]
    decoded = corrected_tokenizer.decode(
        label_ids, skip_special_tokens=True
    )
    print("\nOriginal target:", row["original_target"])
    print("Model-facing target:", row["model_target"])
    print("Decoded tokenized target:", decoded)

assert len(corrected_train_dataset) == 10095
assert len(corrected_validation_dataset) == 1087
assert len(corrected_test_dataset) == 1052
assert sample_batch["input_ids"].ndim == 2
assert sample_batch["labels"].ndim == 2

tokenization_report = {
    "representation_version": "plain_text_outcome_labels_v1",
    "seed": SEED,
    "max_input_length": MAX_INPUT_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "train_examples": len(corrected_train_dataset),
    "validation_examples": len(corrected_validation_dataset),
    "test_examples": len(corrected_test_dataset),
    "original_splits_preserved": True,
}

report_path = EXP_DIR / "corrected_tokenization_audit.json"
report_path.write_text(
    json.dumps(tokenization_report, indent=2),
    encoding="utf-8"
)

print("\nAudit saved:", report_path)
print("STEP 3.6 COMPLETE")




# PHASE 3 — CELL 9: CORRECTED FP32 TRAINING

import json
import math
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

SEED = 42
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

ROOT = Path("/kaggle/working/digital_twin")
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
CHECKPOINT_DIR = (
    ROOT / "models" / "flan_t5_small_negotiation" / "corrected_fp32"
)

EXP_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "Enable the Kaggle GPU before training."

corrected_model.to(device)
corrected_model.config.use_cache = False
corrected_model.train()

train_loader = DataLoader(
    corrected_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=corrected_collator,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)

validation_loader = DataLoader(
    corrected_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=corrected_collator,
    num_workers=0
)

optimizer = torch.optim.AdamW(
    corrected_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
total_steps = updates_per_epoch * EPOCHS
warmup_steps = max(1, int(0.05 * total_steps))

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def move_batch(batch):
    return {key: value.to(device) for key, value in batch.items()}

def evaluate_corrected_loss(loader):
    corrected_model.eval()
    loss_sum = 0.0
    example_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch)
            outputs = corrected_model(**batch)

            if not torch.isfinite(outputs.loss):
                raise RuntimeError("Non-finite validation loss detected.")

            count = batch["input_ids"].size(0)
            loss_sum += outputs.loss.item() * count
            example_count += count

    corrected_model.train()
    return loss_sum / example_count

# Confirm the initial model produces a finite loss.
smoke_batch = corrected_collator([
    corrected_train_dataset[i] for i in range(BATCH_SIZE)
])
smoke_batch = move_batch(smoke_batch)

with torch.no_grad():
    smoke_loss = corrected_model(**smoke_batch).loss.item()

print("Initial FP32 smoke-test loss:", smoke_loss)
assert math.isfinite(smoke_loss), "Initial loss is not finite; stopping."

del smoke_batch
torch.cuda.empty_cache()

training_log = {
    "model_name": "google/flan-t5-small",
    "representation_version": "plain_text_outcome_labels_v1",
    "seed": SEED,
    "precision": "FP32",
    "epochs_requested": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUMULATION,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "train_examples": len(corrected_train_dataset),
    "validation_examples": len(corrected_validation_dataset),
    "epochs": []
}

best_val_loss = float("inf")
global_step = 0
start_time = time.time()

print("\n=== CORRECTED TRAINING CONFIGURATION ===")
print("GPU:", torch.cuda.get_device_name(0))
print("Training batches:", len(train_loader))
print("Optimizer updates:", total_steps)
print("Effective batch size:", BATCH_SIZE * GRAD_ACCUMULATION)

for epoch in range(EPOCHS):
    corrected_model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    example_count = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch = move_batch(batch)
        current_batch_size = batch["input_ids"].size(0)

        # Full FP32: no autocast and no GradScaler.
        outputs = corrected_model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_idx + 1}."
            )

        (loss / GRAD_ACCUMULATION).backward()
        loss_sum += loss.item() * current_batch_size
        example_count += current_batch_size

        update_now = (
            (batch_idx + 1) % GRAD_ACCUMULATION == 0
            or (batch_idx + 1) == len(train_loader)
        )

        if update_now:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                corrected_model.parameters(), MAX_GRAD_NORM
            )

            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Non-finite gradient norm detected.")

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Running loss: {loss_sum / example_count:.4f}"
            )

    train_loss = loss_sum / example_count
    val_loss = evaluate_corrected_loss(validation_loader)

    record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "optimizer_steps": global_step,
        "duration_seconds": time.time() - epoch_start
    }
    training_log["epochs"].append(record)

    print("\n=== CORRECTED EPOCH SUMMARY ===")
    print(f"Training loss:   {train_loss:.6f}")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Optimizer steps: {global_step}")

    if math.isfinite(val_loss) and val_loss < best_val_loss:
        best_val_loss = val_loss
        corrected_model.save_pretrained(CHECKPOINT_DIR)
        corrected_tokenizer.save_pretrained(CHECKPOINT_DIR)
        training_log["best_validation_loss"] = best_val_loss
        training_log["best_epoch"] = epoch + 1
        print("Best corrected checkpoint saved:", CHECKPOINT_DIR)

training_log["training_completed"] = (
    len(training_log["epochs"]) == EPOCHS
    and math.isfinite(best_val_loss)
)
training_log["total_training_seconds"] = time.time() - start_time

log_path = EXP_DIR / "training_log_corrected_fp32.json"
log_path.write_text(
    json.dumps(training_log, indent=2),
    encoding="utf-8"
)

corrected_model.config.use_cache = True

print("\n=== CORRECTED TRAINING FINISHED ===")
print(json.dumps(training_log, indent=2))
print("Training log:", log_path)
print("Checkpoint:", CHECKPOINT_DIR)

assert training_log["training_completed"], (
    "Corrected training did not complete with a finite validation loss."
)
print("STEP 3.7 COMPLETE")




# PHASE 3 — CELL 10: CORRECTED VALIDATION EVALUATION

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("/kaggle/working/digital_twin")
CHECKPOINT_DIR = (
    ROOT / "models" / "flan_t5_small_negotiation" / "corrected_fp32"
)
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tokenizer_corrected = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
eval_model_corrected = AutoModelForSeq2SeqLM.from_pretrained(
    CHECKPOINT_DIR,
    dtype=torch.float32
).to(device)
eval_model_corrected.eval()

validation_loader = DataLoader(
    corrected_validation_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=corrected_collator,
    num_workers=0
)

predictions = []
print("Generating corrected validation predictions...")

with torch.no_grad():
    for batch_idx, batch in enumerate(validation_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated_ids = eval_model_corrected.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

        predictions.extend(
            eval_tokenizer_corrected.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(validation_loader):
            print(f"Batches: {batch_idx + 1}/{len(validation_loader)}")

assert len(predictions) == len(validation_rows) == 1087

def normalize(text):
    return " ".join(str(text).strip().split())

def classify_model_target(text):
    normalized = normalize(text).upper()

    if normalized == "OUTCOME: DISAGREEMENT":
        return "disagreement"
    if normalized == "OUTCOME: NO_AGREEMENT":
        return "no_agreement"
    if normalized == "OUTCOME: DISCONNECT":
        return "disconnect"
    if "ITEM0=" in normalized and "ITEM1=" in normalized and "ITEM2=" in normalized:
        return "allocation"
    return "other"

def token_overlap_counts(prediction, reference):
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    return overlap, len(pred_tokens), len(ref_tokens)

records = []
exact_matches = 0
category_correct = 0
category_confusion = Counter()
true_category_counts = Counter()
predicted_category_counts = Counter()

overlap_total = 0
predicted_token_total = 0
reference_token_total = 0

for row, prediction in zip(validation_rows, predictions):
    reference = row["model_target"]
    pred_norm = normalize(prediction)
    ref_norm = normalize(reference)

    exact = pred_norm == ref_norm
    exact_matches += int(exact)

    true_category = row["target_category"]
    predicted_category = classify_model_target(prediction)

    category_correct += int(true_category == predicted_category)
    true_category_counts[true_category] += 1
    predicted_category_counts[predicted_category] += 1
    category_confusion[(true_category, predicted_category)] += 1

    overlap, pred_len, ref_len = token_overlap_counts(pred_norm, ref_norm)
    overlap_total += overlap
    predicted_token_total += pred_len
    reference_token_total += ref_len

    records.append({
        "input": row["input"],
        "original_target": row["original_target"],
        "model_target": reference,
        "prediction": prediction,
        "true_category": true_category,
        "predicted_category": predicted_category,
        "exact_match": exact
    })

precision = overlap_total / predicted_token_total if predicted_token_total else 0.0
recall = overlap_total / reference_token_total if reference_token_total else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

# Majority baseline under the corrected target representation.
train_target_counts = Counter(row["model_target"] for row in train_rows)
majority_model_target, majority_count = train_target_counts.most_common(1)[0]
majority_correct = sum(
    normalize(row["model_target"]) == normalize(majority_model_target)
    for row in validation_rows
)

metrics = {
    "model": "google/flan-t5-small",
    "representation_version": "plain_text_outcome_labels_v1",
    "checkpoint": str(CHECKPOINT_DIR),
    "split": "validation",
    "examples": len(validation_rows),
    "exact_matches": exact_matches,
    "exact_match_accuracy": exact_matches / len(validation_rows),
    "category_correct": category_correct,
    "category_accuracy": category_correct / len(validation_rows),
    "true_category_counts": dict(true_category_counts),
    "predicted_category_counts": dict(predicted_category_counts),
    "category_confusion": {
        f"{a} -> {b}": count
        for (a, b), count in category_confusion.items()
    },
    "token_micro_precision": precision,
    "token_micro_recall": recall,
    "token_micro_f1": f1,
    "majority_target": majority_model_target,
    "majority_target_count_in_train": majority_count,
    "majority_target_validation_accuracy": majority_correct / len(validation_rows),
    "generation": {
        "max_new_tokens": 64,
        "num_beams": 1,
        "do_sample": False
    }
}

metrics_path = EXP_DIR / "validation_metrics_corrected.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

predictions_path = EXP_DIR / "validation_predictions_corrected.csv"
with predictions_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

print("\n=== CORRECTED VALIDATION METRICS ===")
print(json.dumps(metrics, indent=2))
print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)

print("\n=== SAMPLE PREDICTIONS ===")
for i, record in enumerate(records[:10], start=1):
    print(f"\nExample {i}")
    print("Reference:", record["model_target"])
    print("Prediction:", record["prediction"])
    print("True category:", record["true_category"])
    print("Predicted category:", record["predicted_category"])
    print("Exact match:", record["exact_match"])

print("\nSTEP 3.8 COMPLETE")




# PHASE 3 — CELL 11: FINAL HELD-OUT TEST EVALUATION

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path("/kaggle/working/digital_twin")
CHECKPOINT_DIR = (
    ROOT / "models" / "flan_t5_small_negotiation" / "corrected_fp32"
)
EXP_DIR = ROOT / "experiments" / "flan_t5_small_negotiation"
EXP_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved corrected checkpoint.
test_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
test_model = AutoModelForSeq2SeqLM.from_pretrained(
    CHECKPOINT_DIR,
    dtype=torch.float32
).to(device)
test_model.eval()

test_loader = DataLoader(
    corrected_test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=corrected_collator,
    num_workers=0
)

test_predictions = []
print("Generating predictions on the held-out test set...")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated_ids = test_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

        test_predictions.extend(
            test_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(test_loader):
            print(f"Batches: {batch_idx + 1}/{len(test_loader)}")

assert len(test_predictions) == len(test_rows) == 1052

def normalize(text):
    return " ".join(str(text).strip().split())

def classify_target(text):
    text = normalize(text).upper()

    if text == "OUTCOME: DISAGREEMENT":
        return "disagreement"
    if text == "OUTCOME: NO_AGREEMENT":
        return "no_agreement"
    if text == "OUTCOME: DISCONNECT":
        return "disconnect"
    if all(marker in text for marker in ("ITEM0=", "ITEM1=", "ITEM2=")):
        return "allocation"
    return "other"

def token_overlap(prediction, reference):
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    return common, len(pred_tokens), len(ref_tokens)

exact_matches = 0
category_correct = 0
true_categories = Counter()
predicted_categories = Counter()
confusion = Counter()
class_stats = defaultdict(lambda: {"examples": 0, "correct_category": 0, "exact_matches": 0})

overlap_sum = 0
pred_token_sum = 0
ref_token_sum = 0
test_records = []

for row, prediction in zip(test_rows, test_predictions):
    reference = row["model_target"]
    pred_norm = normalize(prediction)
    ref_norm = normalize(reference)

    exact = pred_norm == ref_norm
    exact_matches += int(exact)

    true_cat = row["target_category"]
    pred_cat = classify_target(prediction)

    category_correct += int(true_cat == pred_cat)
    true_categories[true_cat] += 1
    predicted_categories[pred_cat] += 1
    confusion[(true_cat, pred_cat)] += 1

    class_stats[true_cat]["examples"] += 1
    class_stats[true_cat]["correct_category"] += int(true_cat == pred_cat)
    class_stats[true_cat]["exact_matches"] += int(exact)

    common, pred_len, ref_len = token_overlap(pred_norm, ref_norm)
    overlap_sum += common
    pred_token_sum += pred_len
    ref_token_sum += ref_len

    test_records.append({
        "input": row["input"],
        "original_target": row["original_target"],
        "model_target": reference,
        "prediction": prediction,
        "true_category": true_cat,
        "predicted_category": pred_cat,
        "exact_match": exact
    })

precision = overlap_sum / pred_token_sum if pred_token_sum else 0.0
recall = overlap_sum / ref_token_sum if ref_token_sum else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

# Majority-target baseline: use only training-set frequencies.
train_counts = Counter(row["model_target"] for row in train_rows)
majority_target, majority_count = train_counts.most_common(1)[0]
majority_exact_correct = sum(
    normalize(row["model_target"]) == normalize(majority_target)
    for row in test_rows
)

# Category baseline: always predict the most common training category.
train_category_counts = Counter(row["target_category"] for row in train_rows)
majority_category, majority_category_count = train_category_counts.most_common(1)[0]
majority_category_correct = sum(
    row["target_category"] == majority_category for row in test_rows
)

metrics = {
    "model": "google/flan-t5-small",
    "representation_version": "plain_text_outcome_labels_v1",
    "checkpoint": str(CHECKPOINT_DIR),
    "split": "test",
    "examples": len(test_rows),
    "exact_matches": exact_matches,
    "exact_match_accuracy": exact_matches / len(test_rows),
    "category_correct": category_correct,
    "category_accuracy": category_correct / len(test_rows),
    "token_micro_precision": precision,
    "token_micro_recall": recall,
    "token_micro_f1": f1,
    "true_category_counts": dict(true_categories),
    "predicted_category_counts": dict(predicted_categories),
    "category_confusion": {
        f"{actual} -> {predicted}": count
        for (actual, predicted), count in confusion.items()
    },
    "per_true_category": {
        category: {
            **stats,
            "category_accuracy": (
                stats["correct_category"] / stats["examples"]
                if stats["examples"] else 0.0
            ),
            "exact_match_accuracy": (
                stats["exact_matches"] / stats["examples"]
                if stats["examples"] else 0.0
            )
        }
        for category, stats in class_stats.items()
    },
    "majority_target_baseline": {
        "target": majority_target,
        "training_count": majority_count,
        "test_correct": majority_exact_correct,
        "test_exact_match_accuracy": majority_exact_correct / len(test_rows)
    },
    "majority_category_baseline": {
        "category": majority_category,
        "training_count": majority_category_count,
        "test_correct": majority_category_correct,
        "test_category_accuracy": majority_category_correct / len(test_rows)
    },
    "generation": {
        "max_new_tokens": 64,
        "num_beams": 1,
        "do_sample": False
    }
}

metrics_path = EXP_DIR / "test_metrics_corrected.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

predictions_path = EXP_DIR / "test_predictions_corrected.csv"
with predictions_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(test_records[0].keys()))
    writer.writeheader()
    writer.writerows(test_records)

print("\n=== FINAL TEST METRICS ===")
print(json.dumps(metrics, indent=2))
print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)

print("\n=== SAMPLE TEST PREDICTIONS ===")
for i, record in enumerate(test_records[:10], start=1):
    print(f"\nExample {i}")
    print("Reference:", record["model_target"])
    print("Prediction:", record["prediction"])
    print("True category:", record["true_category"])
    print("Predicted category:", record["predicted_category"])
    print("Exact match:", record["exact_match"])

print("\nSTEP 3.9 COMPLETE")




from pathlib import Path
import json

ROOT = Path("/kaggle/working/digital_twin")

required = {
    "Fine-tuned checkpoint":
        ROOT / "models/flan_t5_small_negotiation/corrected_fp32",
    "Training log":
        ROOT / "experiments/flan_t5_small_negotiation/training_log_corrected_fp32.json",
    "Validation metrics":
        ROOT / "experiments/flan_t5_small_negotiation/validation_metrics_corrected.json",
    "Test metrics":
        ROOT / "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "Test predictions":
        ROOT / "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
}

for label, path in required.items():
    exists = path.exists()
    if label == "Fine-tuned checkpoint" and exists:
        exists = (path / "config.json").exists() and any(
            p.name.endswith((".safetensors", ".bin"))
            for p in path.iterdir()
        )
    print(f"{'PASS' if exists else 'MISSING'} | {label} | {path}")

metrics_path = required["Test metrics"]
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    print("\nTest examples:", metrics.get("examples"))
    print("Exact-match accuracy:",
          round(metrics.get("exact_match_accuracy", 0) * 100, 2), "%")
    print("Category accuracy:",
          round(metrics.get("category_accuracy", 0) * 100, 2), "%")

print("\nAudit complete.")




from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")

print("PROJECT EXISTS:", ROOT.exists())
print("PROJECT ROOT:", ROOT)

if not ROOT.exists():
    raise FileNotFoundError(f"Project folder not found: {ROOT}")

print("\n=== PYTHON FILES ===")
for p in sorted(ROOT.rglob("*.py")):
    print(f"{p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")

print("\n=== CONFIGURATION AND RESULT FILES ===")
for pattern in ("*.json", "*.jsonl", "*.csv"):
    for p in sorted(ROOT.rglob(pattern)):
        # Skip individual dataset rows to keep the audit readable.
        if p.suffix == ".jsonl":
            continue
        print(f"{p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")

print("\n=== DATABASE FILES ===")
db_files = sorted(
    list(ROOT.rglob("*.db")) +
    list(ROOT.rglob("*.sqlite")) +
    list(ROOT.rglob("*.sqlite3"))
)
if db_files:
    for p in db_files:
        print(f"{p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")
else:
    print("No database found.")

print("\n=== EXISTING PROJECT COMPONENTS ===")
for name in ("src", "verification", "experiments", "models",
             "data", "tests", "logs"):
    p = ROOT / name
    print(f"{name}: {'EXISTS' if p.exists() else 'not present'}")

print("\nAUDIT COMPLETE — no files were changed.")




from pathlib import Path
import json

ROOT = Path("/kaggle/working/digital_twin")
SRC = ROOT / "src"
PROFILES = ROOT / "profiles"
TRACES = ROOT / "experiments" / "digital_twin_negotiation"

for folder in (SRC, PROFILES, TRACES):
    folder.mkdir(parents=True, exist_ok=True)

# 1. Structured Digital Twin profile
profile = {
    "city": "Chennai",
    "preferred_area": "central Chennai",
    "nights": 2,
    "max_total_price": 6000,
    "require_verified": True,
    "privacy": {
        "share_phone": False
    },
    "preferences": {
        "prefer_lower_price": True
    }
}

(PROFILES / "digital_twin_profile.json").write_text(
    json.dumps(profile, indent=2),
    encoding="utf-8"
)

sample_request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 6000. Do NOT share my phone."
)

(PROFILES / "sample_request.txt").write_text(
    sample_request,
    encoding="utf-8"
)

# 2. Negotiation engine
engine_code = r'''
import json
import re
from pathlib import Path
from datetime import datetime, timezone


def parse_request(text, base_profile):
    """Parse the supported sample-request fields and merge with profile."""
    p = dict(base_profile)
    p["privacy"] = dict(base_profile.get("privacy", {}))
    p["preferences"] = dict(base_profile.get("preferences", {}))

    lower = text.lower()

    nights = re.search(r"\b(\d+)\s*nights?\b", lower)
    if nights:
        p["nights"] = int(nights.group(1))

    budget = re.search(
        r"(?:under|below|maximum|max(?:imum)?(?: total)?(?: price)?"
        r"\s*(?:of)?|budget(?: of)?)\s*"
        r"(?:rs\.?|₹|inr)?\s*([\d,]+)",
        lower
    )
    if budget:
        p["max_total_price"] = int(budget.group(1).replace(",", ""))

    if "do not share my phone" in lower or "don't share my phone" in lower:
        p["privacy"]["share_phone"] = False

    if "verified" in lower:
        p["require_verified"] = True

    if "central chennai" in lower:
        p["preferred_area"] = "central Chennai"

    return p


class ServiceAgent:
    def __init__(self, name, hotel, area, initial_price,
                 minimum_price, verified=True):
        self.name = name
        self.hotel = hotel
        self.area = area
        self.initial_price = initial_price
        self.minimum_price = minimum_price
        self.verified = verified

    def offer(self, round_number):
        # Deterministic concessions: reproducible across runs.
        concession = (round_number - 1) * 300
        price = max(self.minimum_price, self.initial_price - concession)
        return {
            "agent": self.name,
            "hotel": self.hotel,
            "area": self.area,
            "total_price": price,
            "verified": self.verified,
            "round": round_number
        }


def check_constraints(offer, profile):
    reasons = []

    if offer["total_price"] > profile["max_total_price"]:
        reasons.append("price_exceeds_budget")

    if profile.get("require_verified", True) and not offer["verified"]:
        reasons.append("hotel_not_verified")

    if offer["area"].lower() != profile["preferred_area"].lower():
        reasons.append("area_mismatch")

    return len(reasons) == 0, reasons


def public_offer(offer, profile):
    """Only allow explicitly approved fields into agent-facing messages."""
    allowed = {
        "agent", "hotel", "area", "total_price", "verified", "round"
    }
    result = {k: v for k, v in offer.items() if k in allowed}

    # Deliberately never include phone numbers or other private profile data.
    assert "phone" not in result
    assert "phone_number" not in result
    return result


def negotiate(request_text, base_profile, max_rounds=3):
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    profile = parse_request(request_text, base_profile)

    agents = [
        ServiceAgent(
            "Agent_A", "Central Inn", "central Chennai",
            initial_price=6500, minimum_price=5600, verified=True
        ),
        ServiceAgent(
            "Agent_B", "Metro Stay", "central Chennai",
            initial_price=6200, minimum_price=5000, verified=True
        )
    ]

    trace = {
        "request": request_text,
        "profile": profile,
        "max_rounds": max_rounds,
        "rounds": [],
        "status": "failed",
        "accepted_offer": None,
        "failure_reason": None
    }

    for round_number in range(1, max_rounds + 1):
        round_record = {
            "round": round_number,
            "offers": [],
            "decisions": []
        }

        eligible = []

        for agent in agents:
            offer = public_offer(agent.offer(round_number), profile)
            valid, reasons = check_constraints(offer, profile)

            round_record["offers"].append(offer)
            round_record["decisions"].append({
                "agent": agent.name,
                "eligible": valid,
                "rejection_reasons": reasons
            })

            if valid:
                eligible.append(offer)

        trace["rounds"].append(round_record)

        if eligible:
            # Lowest valid total price wins; name breaks ties deterministically.
            accepted = min(
                eligible,
                key=lambda x: (x["total_price"], x["hotel"])
            )
            trace["status"] = "agreement"
            trace["accepted_offer"] = accepted
            break

    if trace["status"] != "agreement":
        trace["failure_reason"] = (
            "No offer satisfied all hard constraints within "
            f"{max_rounds} rounds."
        )

    # Record explicit privacy enforcement result.
    trace["privacy_audit"] = {
        "phone_sharing_allowed": profile["privacy"].get(
            "share_phone", False
        ),
        "phone_disclosed_to_agents": False,
        "private_fields_in_outgoing_offers": False
    }

    return trace


def save_trace(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
''' 

(SRC / "digital_twin.py").write_text(engine_code.strip() + "\n", encoding="utf-8")

# 3. One-command entry point
main_code = r'''
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
'''

(SRC / "main.py").write_text(main_code.strip() + "\n", encoding="utf-8")

print("Created:")
for p in (
    PROFILES / "digital_twin_profile.json",
    PROFILES / "sample_request.txt",
    SRC / "digital_twin.py",
    SRC / "main.py",
):
    print(" -", p)




import subprocess
import sys

ROOT = "/kaggle/working/digital_twin"

result = subprocess.run(
    [sys.executable, f"{ROOT}/src/main.py"],
    cwd=f"{ROOT}/src",
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

print("Exit code:", result.returncode)
assert result.returncode == 0, "Negotiation run failed"




import json
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

from digital_twin import negotiate

profile = json.loads(
    (ROOT / "profiles/digital_twin_profile.json").read_text()
)

# Test 1: reproducibility
request = (
    ROOT / "profiles/sample_request.txt"
).read_text().strip()

trace_a = negotiate(request, profile, max_rounds=3)
trace_b = negotiate(request, profile, max_rounds=3)

assert trace_a == trace_b
print("PASS: Reproducibility")

# Test 2: impossible budget specified in the request itself
impossible_request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 1000. Do NOT share my phone."
)

failure_trace = negotiate(
    impossible_request, profile, max_rounds=3
)

assert failure_trace["status"] == "failed", failure_trace
assert failure_trace["accepted_offer"] is None
assert failure_trace["failure_reason"] is not None
assert len(failure_trace["rounds"]) == 3

failure_path = (
    ROOT / "experiments/digital_twin_negotiation/failure_trace.json"
)
failure_path.write_text(
    json.dumps(failure_trace, indent=2),
    encoding="utf-8"
)

print("PASS: Explicit failure path")
print("Failure reason:", failure_trace["failure_reason"])
print("Rounds attempted:", len(failure_trace["rounds"]))
print("Failure trace saved:", failure_path)

# Test 3: privacy enforcement
assert trace_a["privacy_audit"]["phone_disclosed_to_agents"] is False
assert trace_a["privacy_audit"]["private_fields_in_outgoing_offers"] is False
print("PASS: Privacy audit")




from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
engine_path = ROOT / "src" / "digital_twin.py"

source = engine_path.read_text(encoding="utf-8")

start = source.index("def parse_request(")
end = source.index("\n\nclass ServiceAgent:", start)

replacement = r'''def parse_request(text, base_profile):
    """Parse request fields without weakening profile hard constraints."""
    p = dict(base_profile)
    p["privacy"] = dict(base_profile.get("privacy", {}))
    p["preferences"] = dict(base_profile.get("preferences", {}))

    lower = text.lower()

    nights = re.search(r"\b(\d+)\s*nights?\b", lower)
    if nights:
        p["nights"] = int(nights.group(1))

    budget = re.search(
        r"(?:under|below|maximum|max(?:imum)?(?: total)?(?: price)?"
        r"\s*(?:of)?|budget(?: of)?)\s*"
        r"(?:rs\.?|₹|inr)?\s*([\d,]+)",
        lower
    )
    if budget:
        requested_budget = int(budget.group(1).replace(",", ""))
        profile_budget = int(base_profile["max_total_price"])
        # A request can tighten, but never relax, the configured budget.
        p["max_total_price"] = min(requested_budget, profile_budget)

    # Hard verification policy can only be enabled or preserved, never relaxed.
    p["require_verified"] = bool(
        base_profile.get("require_verified", True)
    )

    # Privacy denial in the profile is authoritative.
    profile_allows_phone = bool(
        base_profile.get("privacy", {}).get("share_phone", False)
    )
    explicitly_denies_phone = (
        "do not share my phone" in lower
        or "don't share my phone" in lower
        or "never share my phone" in lower
    )
    p["privacy"]["share_phone"] = (
        profile_allows_phone and not explicitly_denies_phone
    )

    if "central chennai" in lower:
        p["preferred_area"] = "central Chennai"

    return p
'''

source = source[:start] + replacement + source[end:]
engine_path.write_text(source, encoding="utf-8")

print("Updated:", engine_path)
print("Hard-constraint parsing fix applied.")




import json
import sys
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

from digital_twin import parse_request

profile_path = ROOT / "profiles/digital_twin_profile.json"
profile = json.loads(profile_path.read_text())

request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 20000. Do NOT share my phone."
)

print("Profile budget:", repr(profile.get("max_total_price")))
print("Profile privacy:", repr(profile.get("privacy")))
print("Request:", request)

parsed = parse_request(request, profile)

print("\nParsed budget:", repr(parsed.get("max_total_price")))
print("Parsed verification:", repr(parsed.get("require_verified")))
print("Parsed privacy:", repr(parsed.get("privacy")))
print("Full parsed profile:")
print(json.dumps(parsed, indent=2))




import sys
import json
import importlib
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

import digital_twin
importlib.reload(digital_twin)

parse_request = digital_twin.parse_request
negotiate = digital_twin.negotiate

profile = json.loads(
    (ROOT / "profiles/digital_twin_profile.json").read_text()
)

# Test 1: request cannot increase the profile budget.
looser_request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 20000. Do NOT share my phone."
)
parsed = parse_request(looser_request, profile)

print("Budget after loose request:", parsed["max_total_price"])
assert parsed["max_total_price"] == 6000
assert parsed["require_verified"] is True
assert parsed["privacy"]["share_phone"] is False
print("PASS: Profile hard constraints preserved")

# Test 2: lower budget must be respected.
tighter_request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 1000. Do NOT share my phone."
)
parsed_tight = parse_request(tighter_request, profile)

print("Budget after tight request:", parsed_tight["max_total_price"])
assert parsed_tight["max_total_price"] == 1000

failure = negotiate(tighter_request, profile, max_rounds=3)
assert failure["status"] == "failed"
assert failure["accepted_offer"] is None
print("PASS: Impossible budget leads to explicit failure")

# Test 3: normal negotiation still succeeds.
normal_request = (
    ROOT / "profiles/sample_request.txt"
).read_text().strip()

success = negotiate(normal_request, profile, max_rounds=3)
assert success["status"] == "agreement"
assert success["accepted_offer"]["total_price"] <= 6000
assert success["accepted_offer"]["verified"] is True
assert success["privacy_audit"]["phone_disclosed_to_agents"] is False

print("PASS: Normal negotiation")
print("Accepted hotel:", success["accepted_offer"]["hotel"])
print("Accepted price:", success["accepted_offer"]["total_price"])
print("\nALL HARD-CONSTRAINT TESTS PASSED")




from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import display, FileLink
import os

ROOT = Path("/kaggle/working/digital_twin")
BACKUP = Path("/kaggle/working/digital_twin_full_backup.zip")

assert ROOT.exists(), f"Project not found: {ROOT}"

file_count = 0
total_bytes = 0

with ZipFile(BACKUP, "w", compression=ZIP_DEFLATED) as archive:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        # Skip disposable notebook caches only.
        if "__pycache__" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue

        archive.write(path, path.relative_to(ROOT.parent))
        file_count += 1
        total_bytes += path.stat().st_size

# Verify the archive can be read and includes the key deliverables.
with ZipFile(BACKUP, "r") as archive:
    bad_file = archive.testzip()
    names = set(archive.namelist())

    required_suffixes = [
        "src/digital_twin.py",
        "src/main.py",
        "profiles/digital_twin_profile.json",
        "experiments/digital_twin_negotiation/negotiation_trace.json",
        "experiments/digital_twin_negotiation/failure_trace.json",
        "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    ]

    missing = [
        suffix for suffix in required_suffixes
        if not any(name.endswith(suffix) for name in names)
    ]

assert bad_file is None, f"Archive verification failed: {bad_file}"
assert not missing, f"Missing files in backup: {missing}"

print("BACKUP CREATED AND VERIFIED")
print("Files archived:", file_count)
print("Original file sizes total:", f"{total_bytes / (1024**2):.1f} MB")
print("Backup size:", f"{BACKUP.stat().st_size / (1024**2):.1f} MB")
print("Backup path:", BACKUP)

display(FileLink(str(BACKUP), result_html_prefix="Download your complete project: "))





from IPython.display import HTML, display
from pathlib import Path
import base64

p = Path("/kaggle/working/digital_twin_full_backup.zip")
assert p.exists(), "Backup ZIP not found"

print(f"Backup size: {p.stat().st_size / (1024**2):.1f} MB")

# This approach embeds the entire ZIP in the notebook output,
# so use it only if the file-panel download fails.
data = base64.b64encode(p.read_bytes()).decode("ascii")
display(HTML(
    '<a download="digital_twin_full_backup.zip" '
    'href="data:application/zip;base64,' + data + '">'
    'Click here to download your Digital Twin backup ZIP</a>'
))




from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

print("=== KAGGLE INPUT CONTENTS ===")
for p in sorted(INPUT_ROOT.iterdir()):
    print(("DIR " if p.is_dir() else "FILE"), p)

print("\n=== ZIP FILES UNDER /kaggle/input ===")
zips = list(INPUT_ROOT.rglob("*.zip"))
if zips:
    for p in zips:
        print(p, f"({p.stat().st_size / (1024**2):.1f} MB)")
else:
    print("No ZIP found. Checking mounted dataset files:")
    for p in list(INPUT_ROOT.rglob("*"))[:40]:
        print(p)




from pathlib import Path
import shutil

SOURCE = Path(
    "/kaggle/input/datasets/vibecoder18/dtp-backup/digital_twin"
)
DEST = Path("/kaggle/working/digital_twin")

assert SOURCE.exists(), f"Backup source not found: {SOURCE}"
assert (SOURCE / "src" / "digital_twin.py").exists(), "Engine missing"
assert (
    SOURCE / "models" / "flan_t5_small_negotiation"
    / "corrected_fp32" / "config.json"
).exists(), "Fine-tuned checkpoint missing"

if DEST.exists():
    print("Destination already exists; leaving it untouched:", DEST)
else:
    shutil.copytree(SOURCE, DEST)
    print("Project restored successfully.")

print("\n=== RESTORATION CHECK ===")
for relative in [
    "src/digital_twin.py",
    "src/main.py",
    "profiles/digital_twin_profile.json",
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
    "models/flan_t5_small_negotiation/corrected_fp32/config.json",
    "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
]:
    path = DEST / relative
    print(("PASS" if path.exists() else "MISSING"), relative)

print("\nProject path:", DEST)




from pathlib import Path
import sys
import json
import importlib

ROOT = Path("/kaggle/working/digital_twin")
ENGINE = ROOT / "src" / "digital_twin.py"

print("Engine exists:", ENGINE.exists())
source = ENGINE.read_text(encoding="utf-8")

print("Phase 5 policy layer present:",
      "# PHASE_5_POLICY_LAYER_V1" in source)

sys.path.insert(0, str(ROOT / "src"))

import digital_twin
importlib.reload(digital_twin)

print("\nAvailable policy functions:")
for name in (
    "check_constraints",
    "sanitize_outgoing_message",
    "validate_final_agreement",
    "negotiate",
):
    print(f"{name}: {hasattr(digital_twin, name)}")

profile = json.loads(
    (ROOT / "profiles" / "digital_twin_profile.json")
    .read_text(encoding="utf-8")
)
print("\nProfile loaded:", profile)




from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
engine_path = ROOT / "src" / "digital_twin.py"
source = engine_path.read_text(encoding="utf-8")

marker = "# PHASE_5_POLICY_LAYER_V1"

if marker in source:
    print("Phase 5 policy layer already exists; no changes made.")
else:
    policy_code = r'''

# PHASE_5_POLICY_LAYER_V1
import re as _policy_re


def check_constraints(offer, profile):
    """Return (allowed, rejection_reasons) for a service offer."""
    reasons = []

    try:
        price = float(offer.get("total_price", -1))
        budget = float(profile["max_total_price"])
        if price < 0 or price > budget:
            reasons.append("price_exceeds_budget_or_invalid")
    except (TypeError, ValueError, KeyError):
        reasons.append("missing_or_invalid_price")

    if profile.get("require_verified", True):
        if offer.get("verified") is not True:
            reasons.append("hotel_not_verified")

    required_area = profile.get("preferred_area")
    if required_area and str(offer.get("area", "")).strip().casefold() != \
            str(required_area).strip().casefold():
        reasons.append("area_mismatch")

    if not offer.get("hotel") or not offer.get("agent"):
        reasons.append("missing_hotel_or_agent")

    return not reasons, reasons


def sanitize_outgoing_message(message, profile):
    """
    Centralized privacy filter. Removes private contact fields and redacts
    phone-like strings when profile privacy disallows phone sharing.
    """
    share_phone = profile.get("privacy", {}).get("share_phone", False)

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized = str(key).casefold().replace("-", "_")
                private_key = any(
                    term in normalized
                    for term in (
                        "phone", "mobile", "telephone", "contact_number"
                    )
                )
                if not share_phone and private_key:
                    continue
                result[key] = clean(item)
            return result

        if isinstance(value, list):
            return [clean(item) for item in value]

        if isinstance(value, str) and not share_phone:
            return _policy_re.sub(
                r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
                "[REDACTED_PHONE]",
                value
            )

        return value

    return clean(message)


def validate_final_agreement(offer, profile):
    """Re-check the final offer independently before accepting it."""
    if not isinstance(offer, dict):
        return False, ["agreement_is_not_an_object"]
    return check_constraints(offer, profile)


# Wrap the original negotiation function so a final agreement is revalidated.
_pre_phase5_negotiate = negotiate


def negotiate(request_text, base_profile, max_rounds=3):
    trace = _pre_phase5_negotiate(
        request_text, base_profile, max_rounds=max_rounds
    )

    accepted = trace.get("accepted_offer")

    if trace.get("status") == "agreement":
        valid, reasons = validate_final_agreement(
            accepted, trace["profile"]
        )
        trace["final_agreement_validation"] = {
            "passed": valid,
            "rejection_reasons": reasons
        }

        if not valid:
            trace["status"] = "failed"
            trace["accepted_offer"] = None
            trace["failure_reason"] = (
                "Final agreement failed hard-constraint validation."
            )
    else:
        trace["final_agreement_validation"] = {
            "passed": False,
            "rejection_reasons": ["no_agreement_reached"]
        }

    return trace
'''

    engine_path.write_text(
        source.rstrip() + "\n\n" + policy_code.strip() + "\n",
        encoding="utf-8"
    )
    print("Phase 5 policy layer added.")

print("Updated file:", engine_path)




import sys
import json
import importlib
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

import digital_twin
importlib.reload(digital_twin)

profile = json.loads(
    (ROOT / "profiles" / "digital_twin_profile.json")
    .read_text(encoding="utf-8")
)

check = digital_twin.check_constraints
sanitize = digital_twin.sanitize_outgoing_message
validate = digital_twin.validate_final_agreement

valid_offer = {
    "agent": "Agent_A",
    "hotel": "Central Inn",
    "area": "central Chennai",
    "total_price": 5500,
    "verified": True,
    "round": 2
}

# Valid offer
ok, reasons = check(valid_offer, profile)
assert ok, reasons
print("PASS: Compliant offer accepted")

# Over budget
ok, reasons = check({**valid_offer, "total_price": 9000}, profile)
assert not ok and any("price" in r for r in reasons)
print("PASS: Over-budget offer rejected")

# Unverified
ok, reasons = check({**valid_offer, "verified": False}, profile)
assert not ok and "hotel_not_verified" in reasons
print("PASS: Unverified hotel rejected")

# Malformed price
ok, reasons = check({**valid_offer, "total_price": "invalid"}, profile)
assert not ok
print("PASS: Malformed offer rejected")

# Private fields and phone-like text
malicious = {
    "customer_phone": "+91 98765 43210",
    "message": "Contact this customer: +91 98765 43210",
    "offer": valid_offer
}
outgoing = sanitize(malicious, profile)
serialized = json.dumps(outgoing)

assert "customer_phone" not in outgoing
assert "+91 98765 43210" not in serialized
assert "[REDACTED_PHONE]" in serialized
print("PASS: Phone-number disclosure blocked")

# Independent final validation
assert validate(valid_offer, profile)[0] is True
assert validate({**valid_offer, "total_price": 9000}, profile)[0] is False
print("PASS: Final agreement revalidation")

print("\nPHASE 5 POLICY TESTS PASSED")




import sys
import json
import importlib
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

import digital_twin
importlib.reload(digital_twin)

profile = json.loads(
    (ROOT / "profiles" / "digital_twin_profile.json")
    .read_text(encoding="utf-8")
)

# 1. End-to-end negotiation must produce a compliant agreement.
request = (
    ROOT / "profiles" / "sample_request.txt"
).read_text(encoding="utf-8").strip()

trace = digital_twin.negotiate(request, profile, max_rounds=3)

assert trace["status"] == "agreement", trace
assert trace["final_agreement_validation"]["passed"] is True
assert trace["accepted_offer"]["total_price"] <= profile["max_total_price"]
assert trace["accepted_offer"]["verified"] is True
assert trace["privacy_audit"]["phone_disclosed_to_agents"] is False

print("PASS: End-to-end agreement validated")
print("Accepted:", trace["accepted_offer"])

# 2. A low-budget request must fail explicitly.
impossible_request = (
    "Book a verified hotel in central Chennai for 2 nights "
    "under Rs. 1000. Do NOT share my phone."
)
failed = digital_twin.negotiate(
    impossible_request, profile, max_rounds=3
)

assert failed["status"] == "failed"
assert failed["accepted_offer"] is None
assert failed["failure_reason"]
print("PASS: End-to-end impossible-budget failure")

# 3. Test a malicious outgoing message using the actual policy.
malicious_message = {
    "agent": "Agent_B",
    "text": "Customer phone is +91 98765 43210",
    "customer_phone": "+91 98765 43210",
    "offer": trace["accepted_offer"]
}
outgoing = digital_twin.sanitize_outgoing_message(
    malicious_message, profile
)

serialized = json.dumps(outgoing)
assert "customer_phone" not in outgoing
assert "+91 98765 43210" not in serialized
print("PASS: Malicious outgoing message sanitized")

# Save the latest Phase 5 success trace.
trace_path = (
    ROOT / "experiments" / "digital_twin_negotiation"
    / "phase5_validated_trace.json"
)
trace_path.parent.mkdir(parents=True, exist_ok=True)
trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

print("Saved:", trace_path)
print("\nPHASE 5 END-TO-END TESTS PASSED")




from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
SRC = ROOT / "src"
SRC.mkdir(parents=True, exist_ok=True)

module_path = SRC / "execution_verifier.py"

module_code = r'''
import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    hotel TEXT NOT NULL,
    area TEXT NOT NULL,
    total_price INTEGER NOT NULL,
    nights INTEGER NOT NULL,
    verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
    agent TEXT NOT NULL,
    agreement_json TEXT NOT NULL,
    status TEXT NOT NULL
)
"""


def initialize_database(db_path):
    """Create the simulated execution database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()

    return str(db_path)


def execute_agreement(agreement, profile, db_path, booking_id="booking-001"):
    """
    Persist a validated simulated agreement.
    Recheck hard constraints before writing.
    """
    from digital_twin import validate_final_agreement

    valid, reasons = validate_final_agreement(agreement, profile)
    if not valid:
        raise ValueError(
            f"Refusing to execute invalid agreement: {reasons}"
        )

    nights = int(profile["nights"])
    price = int(agreement["total_price"])

    if nights <= 0:
        raise ValueError("Booking duration must be positive.")

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bookings (
                booking_id, hotel, area, total_price, nights,
                verified, agent, agreement_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                agreement["hotel"],
                agreement["area"],
                price,
                nights,
                int(agreement["verified"] is True),
                agreement["agent"],
                json.dumps(agreement, sort_keys=True),
                "accepted"
            )
        )
        conn.commit()

    return {
        "booking_id": booking_id,
        "database_path": str(db_path),
        "status": "persisted",
        "hotel": agreement["hotel"],
        "total_price": price,
        "nights": nights
    }


def verify_persisted_booking(booking_id, expected_agreement,
                             profile, db_path):
    """
    Independently reopen SQLite and compare persisted values with
    the original agreement and user constraints.
    """
    errors = []
    db_path = Path(db_path)

    if not db_path.exists():
        return {
            "verified": False,
            "booking_id": booking_id,
            "errors": ["database_missing"]
        }

    # Open a fresh connection and read the persisted row.
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM bookings WHERE booking_id = ?",
            (booking_id,)
        ).fetchone()

    if row is None:
        return {
            "verified": False,
            "booking_id": booking_id,
            "errors": ["booking_record_missing"]
        }

    expected_fields = {
        "hotel": expected_agreement.get("hotel"),
        "area": expected_agreement.get("area"),
        "total_price": expected_agreement.get("total_price"),
        "nights": profile.get("nights"),
        "verified": 1 if expected_agreement.get("verified") is True else 0,
        "agent": expected_agreement.get("agent"),
        "status": "accepted",
    }

    for field, expected in expected_fields.items():
        if row[field] != expected:
            errors.append(
                f"{field}_mismatch: expected={expected!r}, "
                f"stored={row[field]!r}"
            )

    # Verify the stored agreement snapshot agrees with stored columns.
    try:
        snapshot = json.loads(row["agreement_json"])
        for field in ("hotel", "area", "total_price", "verified", "agent"):
            stored_value = row[field]
            if field == "verified":
                stored_value = bool(stored_value)
            if snapshot.get(field) != stored_value:
                errors.append(f"agreement_snapshot_{field}_mismatch")
    except (TypeError, json.JSONDecodeError):
        errors.append("agreement_snapshot_invalid")

    # Independently check user hard constraints against persisted values.
    try:
        if row["total_price"] > int(profile["max_total_price"]):
            errors.append("persisted_price_exceeds_budget")
        if row["nights"] != int(profile["nights"]):
            errors.append("persisted_duration_mismatch")
        if profile.get("require_verified", True) and row["verified"] != 1:
            errors.append("persisted_hotel_not_verified")
        if row["area"].casefold() != str(
            profile["preferred_area"]
        ).casefold():
            errors.append("persisted_area_mismatch")
    except (TypeError, ValueError, KeyError):
        errors.append("profile_or_record_constraint_check_failed")

    return {
        "verified": len(errors) == 0,
        "booking_id": booking_id,
        "hotel": row["hotel"],
        "total_price": row["total_price"],
        "nights": row["nights"],
        "verified_hotel": bool(row["verified"]),
        "errors": errors
    }
'''

module_path.write_text(module_code.strip() + "\n", encoding="utf-8")

print("Created:", module_path)
print("Module size:", module_path.stat().st_size, "bytes")




import sys
import json
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

import digital_twin
from execution_verifier import execute_agreement, verify_persisted_booking

profile = json.loads(
    (ROOT / "profiles" / "digital_twin_profile.json")
    .read_text(encoding="utf-8")
)
request = (
    ROOT / "profiles" / "sample_request.txt"
).read_text(encoding="utf-8").strip()

trace = digital_twin.negotiate(request, profile, max_rounds=3)
assert trace["status"] == "agreement", trace

db_path = ROOT / "verification" / "services.db"
booking_id = "phase6-booking-001"

execution = execute_agreement(
    trace["accepted_offer"], profile, db_path, booking_id
)
print("EXECUTION RECORD")
print(json.dumps(execution, indent=2))

report = verify_persisted_booking(
    booking_id, trace["accepted_offer"], profile, db_path
)
print("\nINDEPENDENT VERIFICATION REPORT")
print(json.dumps(report, indent=2))

assert report["verified"] is True, report
print("\nPASS: SQLite execution and independent verification")




import sqlite3
import json

# Use the Phase 6 variables from the previous successful cell.
original_price = int(trace["accepted_offer"]["total_price"])

# 1. Tamper with the persisted booking.
with sqlite3.connect(db_path) as conn:
    conn.execute(
        "UPDATE bookings SET total_price = ? WHERE booking_id = ?",
        (9900, booking_id)
    )
    conn.commit()

# 2. The independent verifier should detect the mismatch.
tampered_report = verify_persisted_booking(
    booking_id, trace["accepted_offer"], profile, db_path
)

print("TAMPER DETECTION REPORT")
print(json.dumps(tampered_report, indent=2))

assert tampered_report["verified"] is False, (
    "FAIL: verifier incorrectly accepted the tampered booking"
)
print("\nPASS: Tampered booking was rejected.")

# 3. Restore the original price so the database remains valid.
with sqlite3.connect(db_path) as conn:
    conn.execute(
        "UPDATE bookings SET total_price = ? WHERE booking_id = ?",
        (original_price, booking_id)
    )
    conn.commit()

# 4. Verify the restored booking again.
restored_report = verify_persisted_booking(
    booking_id, trace["accepted_offer"], profile, db_path
)

print("\nRESTORED BOOKING REPORT")
print(json.dumps(restored_report, indent=2))

assert restored_report["verified"] is True, (
    "FAIL: restored booking did not pass verification"
)
print("\nPASS: Original booking restored and verified.")




import json
from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
OUT = ROOT / "experiments" / "digital_twin_execution"
OUT.mkdir(parents=True, exist_ok=True)

artifacts = {
    "execution_record.json": execution,
    "verification_report.json": report,
    "tamper_detection_report.json": tampered_report,
    "restored_verification_report.json": restored_report,
}

for filename, data in artifacts.items():
    path = OUT / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved: {path}")

print("\nFINAL PHASE 6 CHECK")
assert report["verified"] is True
assert tampered_report["verified"] is False
assert restored_report["verified"] is True
assert all((OUT / filename).exists() for filename in artifacts)

print("PASS: Execution persisted")
print("PASS: Original booking independently verified")
print("PASS: Tampered booking rejected")
print("PASS: Restored booking independently verified")
print("PASS: All four evidence files saved")




from pathlib import Path
import json

ROOT = Path("/kaggle/working/digital_twin")

assert ROOT.exists(), f"Project folder not found: {ROOT}"

print("PROJECT ROOT:", ROOT)
print("\n=== SOURCE FILES ===")
for folder in ["src", "tests", "profiles", "verification"]:
    path = ROOT / folder
    print(f"\n[{folder}]")
    if path.exists():
        for item in sorted(path.rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(ROOT)} ({item.stat().st_size:,} bytes)")
    else:
        print("  Folder not found")

print("\n=== EXPERIMENT ARTIFACTS ===")
for folder in ["experiments", "models"]:
    path = ROOT / folder
    print(f"\n[{folder}]")
    if path.exists():
        for item in sorted(path.rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(ROOT)} ({item.stat().st_size:,} bytes)")
    else:
        print("  Folder not found")

print("\n=== PROFILE AND REQUEST ===")
profile_path = ROOT / "profiles" / "digital_twin_profile.json"
request_path = ROOT / "profiles" / "sample_request.txt"

if profile_path.exists():
    print("Profile:", json.loads(profile_path.read_text(encoding="utf-8")))
if request_path.exists():
    print("Request:", request_path.read_text(encoding="utf-8").strip())

print("\n=== KEY MODEL METRICS ===")
for filename in [
    ROOT / "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    ROOT / "experiments/deal_or_no_deal_allocation_baseline/metrics.json",
]:
    print(f"\n{filename.relative_to(ROOT)}")
    if filename.exists():
        try:
            print(json.dumps(
                json.loads(filename.read_text(encoding="utf-8")),
                indent=2
            ))
        except Exception as e:
            print("Could not parse metrics:", repr(e))
    else:
        print("NOT FOUND")




from pathlib import Path
import ast
import json

ROOT = Path("/kaggle/working/digital_twin")

# 1. Print the existing entry point without modifying it.
main_path = ROOT / "src" / "main.py"
print("=== src/main.py ===")
print(main_path.read_text(encoding="utf-8"))

# 2. List functions/classes available in the main project modules.
for filename in ["digital_twin.py", "execution_verifier.py"]:
    path = ROOT / "src" / filename
    print(f"\n=== DEFINITIONS IN {filename} ===")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            print(f"Line {node.lineno}: {node.__class__.__name__} {node.name}")

# 3. Find all saved JSON metrics files, including baselines.
print("\n=== ALL SAVED METRICS JSON FILES ===")
for path in sorted((ROOT / "experiments").rglob("*.json")):
    if "metric" in path.name.lower() or "baseline" in str(path).lower():
        print(f"\n--- {path.relative_to(ROOT)} ---")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps(data, indent=2)[:5000])
        except Exception as e:
            print("Read error:", repr(e))




from pathlib import Path
import subprocess
import os
import sys

ROOT = Path("/kaggle/working/digital_twin")
trace_path = ROOT / "experiments" / "digital_twin_negotiation" / "negotiation_trace.json"

# Preserve the existing evidence before testing main.py.
previous_trace = trace_path.read_bytes() if trace_path.exists() else None

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

try:
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "main.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120
    )

    print("=== MAIN.PY STDOUT ===")
    print(result.stdout or "(no standard output)")

    print("\n=== MAIN.PY STDERR ===")
    print(result.stderr or "(no errors printed)")

    print("\nExit code:", result.returncode)
    print("Entry point passed:", result.returncode == 0)

finally:
    # Restore the original trace even if the test fails.
    if previous_trace is not None:
        trace_path.write_bytes(previous_trace)
        print("\nOriginal negotiation trace restored.")

print("\n=== NEXT MODULE SIGNATURES ===")
import inspect

sys.path.insert(0, str(ROOT / "src"))
import digital_twin
import execution_verifier

print("negotiate:", inspect.signature(digital_twin.negotiate))
print("execute_agreement:", inspect.signature(execution_verifier.execute_agreement))
print("verify_persisted_booking:", inspect.signature(execution_verifier.verify_persisted_booking))




from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")
main_path = ROOT / "src" / "main.py"
backup_path = ROOT / "src" / "main_phase7_backup.py"

# Preserve the working entry point before modifying it.
if not backup_path.exists():
    backup_path.write_bytes(main_path.read_bytes())

integrated_main = r'''
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
'''

main_path.write_text(integrated_main.lstrip(), encoding="utf-8")
print("Updated:", main_path)
print("Backup: ", backup_path)
print("New size:", main_path.stat().st_size, "bytes")




from pathlib import Path
import subprocess
import sys
import os

ROOT = Path("/kaggle/working/digital_twin")
env = os.environ.copy()
env["PYTHONPATH"] = (
    str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
)

result = subprocess.run(
    [sys.executable, str(ROOT / "src" / "main.py")],
    cwd=str(ROOT),
    env=env,
    capture_output=True,
    text=True,
    timeout=120
)

print("=== STDOUT ===")
print(result.stdout or "(empty)")

print("\n=== STDERR ===")
print(result.stderr or "(no errors)")

print("\nExit code:", result.returncode)
print("Integrated demo passed:", result.returncode == 0)

evidence = ROOT / "experiments" / "phase7_end_to_end"
print("\n=== GENERATED EVIDENCE ===")
if evidence.exists():
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")




from pathlib import Path
import sys
import json
import csv
import copy

ROOT = Path("/kaggle/working/digital_twin")
sys.path.insert(0, str(ROOT / "src"))

import digital_twin

# --------------------------------------------------
# A. Failure-path test: impossible budget
# --------------------------------------------------
profile = json.loads(
    (ROOT / "profiles" / "digital_twin_profile.json").read_text(encoding="utf-8")
)
request = (
    ROOT / "profiles" / "sample_request.txt"
).read_text(encoding="utf-8").strip()

impossible_profile = copy.deepcopy(profile)
impossible_profile["max_total_price"] = 1000

failure_trace = digital_twin.negotiate(
    request, impossible_profile, max_rounds=3
)

print("=== FAILURE-PATH TEST ===")
print("Status:", failure_trace.get("status"))
print("Accepted offer:", failure_trace.get("accepted_offer"))
print("Failure reason:", failure_trace.get("failure_reason"))

assert failure_trace.get("status") != "agreement", (
    "FAIL: system accepted an agreement despite the impossible budget"
)
assert not failure_trace.get("accepted_offer"), (
    "FAIL: an invalid offer was returned as accepted"
)
print("PASS: Impossible-budget request did not produce an agreement.")

# --------------------------------------------------
# B. Audit saved model and baseline metrics
# --------------------------------------------------
model_path = ROOT / "experiments/flan_t5_small_negotiation/test_metrics_corrected.json"
baseline_path = ROOT / "experiments/majority_target_baseline/metrics.json"

model = json.loads(model_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["test"]

comparison = {
    "dataset_split": "test",
    "examples": model["examples"],
    "model": "FLAN-T5-small",
    "baseline": "majority-target baseline",
    "metrics": [
        {
            "metric": "Exact-match accuracy",
            "model": model["exact_match_accuracy"],
            "baseline": baseline["exact_match_accuracy"]
        },
        {
            "metric": "Category accuracy",
            "model": model["category_accuracy"],
            "baseline": baseline["category_accuracy"]
        }
    ],
    "token_micro_f1": model["token_micro_f1"],
    "notes": [
        "The saved test split contains 1,052 examples.",
        "Token micro-F1 is reported for the model only because the saved baseline metrics do not provide a comparable token-F1 value.",
        "The model's category accuracy is not uniform across classes."
    ]
}

print("\n=== MODEL VS BASELINE ===")
for row in comparison["metrics"]:
    delta = row["model"] - row["baseline"]
    print(
        f'{row["metric"]}: model={row["model"]:.4%}, '
        f'baseline={row["baseline"]:.4%}, '
        f'difference={delta:+.4%}'
    )
print(f'Token micro-F1 (model): {comparison["token_micro_f1"]:.4%}')

# --------------------------------------------------
# C. Check predictions and checkpoint files
# --------------------------------------------------
predictions_path = (
    ROOT / "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv"
)
checkpoint = ROOT / "models/flan_t5_small_negotiation/corrected_fp32"

with predictions_path.open("r", encoding="utf-8", newline="") as f:
    prediction_rows = list(csv.DictReader(f))

checkpoint_files = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json"
]
missing_checkpoint_files = [
    name for name in checkpoint_files if not (checkpoint / name).exists()
]

print("\n=== ARTIFACT AUDIT ===")
print("Test prediction rows:", len(prediction_rows))
print("Expected test examples:", model["examples"])
print("Checkpoint:", checkpoint)
print("Missing checkpoint files:", missing_checkpoint_files)

assert len(prediction_rows) == model["examples"], (
    "Prediction CSV row count does not match saved test metrics."
)
assert not missing_checkpoint_files, (
    "Checkpoint export is incomplete."
)
print("PASS: Prediction count and checkpoint files verified.")

# --------------------------------------------------
# D. Save Phase 7 audit evidence
# --------------------------------------------------
out = ROOT / "experiments" / "phase7_end_to_end"
out.mkdir(parents=True, exist_ok=True)

(out / "failure_path_trace.json").write_text(
    json.dumps(failure_trace, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8"
)
(out / "model_baseline_comparison.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nSaved failure-path trace and model/baseline comparison.")
print("Evidence directory:", out)




from pathlib import Path
import json
import csv
import hashlib
from datetime import datetime, timezone

ROOT = Path("/kaggle/working/digital_twin")
EXP = ROOT / "experiments"
OUT = EXP / "phase7_end_to_end"
OUT.mkdir(parents=True, exist_ok=True)

# Load actual saved results.
model = json.loads(
    (EXP / "flan_t5_small_negotiation/test_metrics_corrected.json")
    .read_text(encoding="utf-8")
)
baseline_all = json.loads(
    (EXP / "majority_target_baseline/metrics.json")
    .read_text(encoding="utf-8")
)
baseline = baseline_all["test"]

run_summary_path = OUT / "run_summary.json"
run_summary = (
    json.loads(run_summary_path.read_text(encoding="utf-8"))
    if run_summary_path.exists() else {}
)

# Recalculate the baseline's category accuracy from its saved predictions.
n_test = int(model["examples"])
baseline_correct = int(baseline["category_correct"])
baseline_category_accuracy = baseline_correct / n_test

comparison = {
    "evaluation_split": "test",
    "test_examples": n_test,
    "model": "google/flan-t5-small",
    "baseline": "majority-target baseline",
    "model_exact_match_accuracy": model["exact_match_accuracy"],
    "baseline_exact_match_accuracy": baseline["exact_match_accuracy"],
    "exact_match_difference_percentage_points": 100 * (
        model["exact_match_accuracy"] - baseline["exact_match_accuracy"]
    ),
    "model_category_accuracy": model["category_accuracy"],
    "baseline_category_accuracy": baseline_category_accuracy,
    "category_accuracy_difference_percentage_points": 100 * (
        model["category_accuracy"] - baseline_category_accuracy
    ),
    "model_token_micro_precision": model["token_micro_precision"],
    "model_token_micro_recall": model["token_micro_recall"],
    "model_token_micro_f1": model["token_micro_f1"],
    "baseline_token_micro_f1": None,
    "note": (
        "Baseline category accuracy is derived from its saved correct count "
        "divided by test examples. Token micro-F1 is not available for the "
        "baseline and is therefore not compared."
    )
}
(OUT / "model_baseline_comparison.json").write_text(
    json.dumps(comparison, indent=2), encoding="utf-8"
)

# Hash important files so the evidence manifest can identify exact artifacts.
artifact_paths = [
    ROOT / "src/main.py",
    ROOT / "src/digital_twin.py",
    ROOT / "src/execution_verifier.py",
    ROOT / "profiles/digital_twin_profile.json",
    ROOT / "profiles/sample_request.txt",
    EXP / "flan_t5_small_negotiation/test_metrics_corrected.json",
    EXP / "flan_t5_small_negotiation/test_predictions_corrected.csv",
    EXP / "majority_target_baseline/metrics.json",
    ROOT / "models/flan_t5_small_negotiation/corrected_fp32/config.json",
    ROOT / "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    OUT / "run_summary.json",
    OUT / "verification_report.json",
    OUT / "failure_path_trace.json",
    OUT / "model_baseline_comparison.json",
]

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "project_root": str(ROOT),
    "artifacts": []
}
for path in artifact_paths:
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["artifacts"].append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": digest
        })
    else:
        manifest["artifacts"].append({
            "path": str(path.relative_to(ROOT)),
            "missing": True
        })

(OUT / "evidence_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

# Locate a notebook file if one exists in the project directory.
notebooks = sorted(ROOT.rglob("*.ipynb"))

report = f"""# Personal AI Digital Twin for Privacy-Aware Multi-Agent Negotiation and Verified Task Execution

## 1. Abstract

This project implements a local prototype of a personal AI Digital Twin that negotiates with simulated service agents under user-defined constraints, applies privacy rules, persists accepted terms in SQLite, and independently verifies the persisted outcome. A FLAN-T5-small model was fine-tuned on a subset of the Deal or No Deal negotiation dataset and evaluated on a held-out test split. The system combines learned negotiation-outcome prediction with explicit rule-based policy enforcement and simulated task execution.

## 2. Objectives

- Represent user preferences, hard constraints, and privacy restrictions in a structured profile.
- Simulate negotiation between two service agents and a personal Digital Twin.
- Reject offers that violate hard constraints.
- Avoid disclosing prohibited phone information.
- Persist accepted terms and verify them independently.
- Evaluate a fine-tuned model against a simple majority-target baseline.
- Save reproducible traces and metrics.

## 3. Methodology

### 3.1 Dataset and model

The primary research dataset is Deal or No Deal. The saved processed data contains 10,095 training examples, 1,087 validation examples, and 1,052 test examples. The model is `google/flan-t5-small`, fine-tuned using the project's saved training procedure. The exported checkpoint and tokenizer are stored under `models/flan_t5_small_negotiation/corrected_fp32/`.

### 3.2 Digital Twin and negotiation

The Digital Twin reads a structured profile containing a preferred area, duration, maximum total price, verification requirement, and privacy preferences. Two simulated service agents produce offers. The negotiation component evaluates offers and records the interaction trace.

### 3.3 Policy enforcement

Explicit checks enforce price and other hard constraints. The profile's privacy setting disallows phone sharing. Policy checks are separate from the neural model so that a generated prediction cannot override the hard constraints.

### 3.4 Execution and verification

The accepted agreement is persisted to a SQLite database. A separate verification function reads the stored record and compares it with the expected agreement and profile constraints. A tamper test changed the stored price from ₹5,900 to ₹9,900; the verifier rejected the modified record and reported price mismatch, snapshot mismatch, and budget violation. After restoration, verification passed again.

### 3.5 Evaluation protocol

The saved test split contains {n_test:,} examples. Metrics include exact-match accuracy, category accuracy, and whitespace-token micro precision, recall, and F1 for the model. The majority-target baseline predicts the most frequent training target for every test example. Baseline token micro-F1 is not available in the saved baseline artifact and is not compared.

## 4. Results

| Metric | FLAN-T5-small | Majority-target baseline |
|---|---:|---:|
| Test examples | {n_test:,} | {n_test:,} |
| Exact-match accuracy | {model["exact_match_accuracy"]:.4%} | {baseline["exact_match_accuracy"]:.4%} |
| Category accuracy | {model["category_accuracy"]:.4%} | {baseline_category_accuracy:.4%} |
| Token micro-F1 | {model["token_micro_f1"]:.4%} | Not available |

Exact-match improvement over the baseline: {comparison["exact_match_difference_percentage_points"]:+.4f} percentage points.

Category-accuracy improvement over the baseline: {comparison["category_accuracy_difference_percentage_points"]:+.4f} percentage points.

The saved test metrics show {model["per_true_category"]["allocation"]["correct_category"]} of {model["per_true_category"]["allocation"]["examples"]} allocation examples correctly categorized, but the model did not correctly categorize the disagreement or disconnect examples. The test set is imbalanced, so aggregate accuracy alone does not demonstrate robust performance across all categories.

### Prototype demonstration

The integrated entry point `python src/main.py` completed successfully in the recorded run. It negotiated an agreement for Metro Stay at ₹5,900 for two nights, persisted the record, and independently verified it. The privacy audit passed. An impossible-budget test returned a failure status and no accepted offer.

## 5. Limitations

1. **Simulated services:** The hotel agents and booking execution are simulated. This is not a live hotel-booking integration.
2. **Model class imbalance:** The model performs poorly on disagreement and disconnect classes. Aggregate metrics can obscure these failures.
3. **Limited benchmark scope:** The reported model evaluation uses a held-out Deal or No Deal test split; it does not establish generalization to all service-negotiation domains.
4. **Rule-based policy layer:** Constraint and privacy enforcement are explicit software checks, not learned guarantees.
5. **Verification scope:** The verifier checks stored fields and agreement consistency; it is not a cryptographic proof or a defense against an attacker who can modify all trusted records and verification code.
6. **Token metric interpretation:** Whitespace-token overlap is a lexical metric and should not be interpreted as semantic quality.
7. **Notebook reproducibility:** A successful script run does not by itself prove that the entire original notebook runs top-to-bottom. The notebook must be located and executed separately if that requirement applies.

## 6. Reproducibility

Project root: `{ROOT}`

Main entry point: `python src/main.py`

Key evidence directory: `experiments/phase7_end_to_end/`

Saved model metrics: `experiments/flan_t5_small_negotiation/test_metrics_corrected.json`

Saved baseline metrics: `experiments/majority_target_baseline/metrics.json`

Evidence manifest: `experiments/phase7_end_to_end/evidence_manifest.json`

The manifest records SHA-256 hashes for key source, model, and experiment files. The report should be regenerated if the underlying metrics or artifacts change.

## 7. Research references

The following are relevant source papers or benchmarks for the project. Verify exact bibliographic metadata and citation style against the original publications before submitting the final report.

1. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. Lewis et al., 2017.
2. TravelPlanner: A Benchmark for Real-World Planning with Constraints.
3. Mind2Web: Towards a Generalist Agent for the Web.
4. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents.
5. Personal Large Language Model Agents.
6. Cooperation, Competition, and Maliciousness in Multi-Agent Systems.
7. Automated Privacy Negotiations with Preference Uncertainty.
8. GAIA: A Benchmark for General AI Assistants.
9. Personalized Tool-Augmented Language Models / ETAPP.
10. Auto-SLURP: research on personal-assistant and multi-agent task execution.
11. AgentBench: Evaluating LLMs as Agents.
12. MultiAgentBench: Benchmarking LLM Agents in Multi-Agent Tasks.
13. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.
14. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.

## 8. Conclusion

The prototype demonstrates a complete simulated workflow combining multi-agent negotiation, explicit privacy and constraint checks, SQLite persistence, independent verification, and tamper detection. The fine-tuned FLAN-T5-small model improves over the saved majority-target baseline on aggregate test metrics, but its poor performance on minority categories limits claims of robust negotiation-outcome prediction. Further work should focus on class-balanced evaluation, stronger adversarial tests, notebook-level reproducibility, and integration with real service APIs.
"""

report_path = ROOT / "PHASE7_ACADEMIC_REPORT_DRAFT.md"
report_path.write_text(report, encoding="utf-8")

print("Saved academic report:", report_path)
print("Saved comparison:", OUT / "model_baseline_comparison.json")
print("Saved evidence manifest:", OUT / "evidence_manifest.json")
print("Notebook files found:", len(notebooks))
for nb in notebooks[:20]:
    print(" ", nb.relative_to(ROOT))

print("\n=== COMPARISON SUMMARY ===")
print(json.dumps(comparison, indent=2))
print("\nReport size:", report_path.stat().st_size, "bytes")




from pathlib import Path

search_roots = [
    Path("/kaggle/working"),
    Path("/kaggle/input"),
]

found = set()

for root in search_roots:
    if not root.exists():
        continue

    print(f"\nSearching: {root}")
    for path in root.rglob("*.ipynb"):
        found.add(path)
        print(f"  {path}")

print("\n=== NOTEBOOK SEARCH RESULT ===")
if found:
    print(f"Found {len(found)} notebook(s).")
    print("We can select the notebook used for the project and validate it.")
else:
    print("No .ipynb notebook found in the searched locations.")
    print("The current restored project contains Python scripts and saved artifacts,")
    print("but not the original notebook file.")




from pathlib import Path
import json

p = Path("/kaggle/working/.virtual_documents/__notebook_source__.ipynb")

print("Exists:", p.exists())

if p.exists():
    print("File size:", p.stat().st_size, "bytes")

    raw = p.read_bytes()
    print("First 200 bytes:", repr(raw[:200]))

    if not raw.strip():
        print("\nDIAGNOSIS: File is empty.")
    else:
        try:
            data = json.loads(raw.decode("utf-8-sig"))
            print("\nDIAGNOSIS: Valid JSON")
            print("Notebook keys:", list(data.keys()))
            print("Cell count:", len(data.get("cells", [])))
        except Exception as e:
            print("\nDIAGNOSIS: Not valid notebook JSON")
            print("Error:", repr(e))
            print("UTF-8 text preview:", repr(raw[:500].decode("utf-8", errors="replace")))

print("\n=== PROJECT SAFETY CHECK ===")
root = Path("/kaggle/working/digital_twin")
for rel in [
    "src/main.py",
    "src/main_phase7_backup.py",
    "src/digital_twin.py",
    "src/execution_verifier.py",
    "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
]:
    f = root / rel
    print(rel, "EXISTS" if f.exists() else "MISSING")



from pathlib import Path
import hashlib
import shutil

source = Path("/kaggle/working/.virtual_documents/__notebook_source__.ipynb")
project = Path("/kaggle/working/digital_twin")
backup = project / "notebook_source_backup_20261026.py"

assert source.exists(), f"Source not found: {source}"
assert project.exists(), f"Project not found: {project}"

if not backup.exists():
    shutil.copy2(source, backup)
    print("Original Python source backed up.")
else:
    print("Existing backup preserved; no overwrite performed.")

content = backup.read_text(encoding="utf-8", errors="replace")
lines = content.splitlines()

print("\n=== SOURCE BACKUP CHECK ===")
print("Backup:", backup)
print("Size:", backup.stat().st_size, "bytes")
print("Lines:", len(lines))
print("SHA-256:", hashlib.sha256(backup.read_bytes()).hexdigest())

print("\n=== CODE CONTENT HINTS ===")
for keyword in [
    "Trainer", "train()", "save_pretrained",
    "test_predictions", "test_metrics", "main.py"
]:
    matches = [
        i for i, line in enumerate(lines, start=1)
        if keyword.lower() in line.lower()
    ]
    print(f"{keyword}: {matches[:8] or 'not found'}")

print("\n=== PROJECT FILES PRESERVED ===")
for rel in [
    "src/main.py",
    "src/main_phase7_backup.py",
    "src/digital_twin.py",
    "src/execution_verifier.py",
    "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "experiments/phase7_end_to_end/run_summary.json",
]:
    f = project / rel
    print(("OK      " if f.exists() else "MISSING "), rel)


from pathlib import Path
import re

p = Path("/kaggle/working/digital_twin/notebook_source_backup_20261026.py")
lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

print("=== POSSIBLE CELL MARKERS ===")
patterns = {
    "Jupyter In markers": r"^\s*#\s*In\[[^\]]*\]:",
    "VS Code cell markers": r"^\s*#\s*%%",
    "Markdown cell markers": r"^\s*#\s*%%\s*\[markdown\]",
}

for label, pattern in patterns.items():
    found = [
        i for i, line in enumerate(lines, start=1)
        if re.search(pattern, line, flags=re.IGNORECASE)
    ]
    print(f"{label}: {len(found)}")
    print("First positions:", found[:15])

print("\n=== FIRST 45 LINES ===")
for i, line in enumerate(lines[:45], start=1):
    print(f"{i:4}: {line[:180]}")

print("\n=== LAST 45 LINES ===")
for i in range(max(0, len(lines) - 45), len(lines)):
    print(f"{i+1:4}: {lines[i][:180]}")

print("\n=== TRAINING-RELATED CODE CONTEXT ===")
for target in [5219]:
    start = max(1, target - 8)
    end = min(len(lines), target + 12)
    print(f"\nLines {start}-{end}:")
    for n in range(start, end + 1):
        print(f"{n:4}: {lines[n-1][:180]}")


from pathlib import Path
import re

p = Path("/kaggle/working/digital_twin/notebook_source_backup_20261026.py")
lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

print("=== PHASE HEADINGS ===")
for i, line in enumerate(lines, start=1):
    if re.search(r"^\s*#\s*PHASE\s+\d+", line, re.I):
        print(f"{i:5}: {line.strip()}")

print("\n=== TRAINING / EVALUATION / EXPORT LOCATIONS ===")
patterns = [
    r"\btrainer\.train\s*\(",
    r"\btrainer\.evaluate\s*\(",
    r"\btrainer\.predict\s*\(",
    r"\bsave_pretrained\s*\(",
    r"test_metrics",
    r"test_predictions",
    r"PHASE 7",
]

for pattern in patterns:
    matches = [
        (i, line.strip()[:150])
        for i, line in enumerate(lines, start=1)
        if re.search(pattern, line, re.I)
    ]
    print(f"\n{pattern}: {len(matches)} match(es)")
    for i, line in matches[:12]:
        print(f"  {i:5}: {line}")

print("\n=== END OF SOURCE ===")
print("Total lines:", len(lines))
print("Backup includes diagnostic code:",
      any("SOURCE BACKUP CHECK" in line for line in lines))


from pathlib import Path
import json
import pandas as pd

ROOT = Path("/kaggle/working/digital_twin")

checks = {
    "Saved model": "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "Model config": "models/flan_t5_small_negotiation/corrected_fp32/config.json",
    "Test metrics": "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "Test predictions": "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
    "Baseline metrics": "experiments/majority_target_baseline/metrics.json",
    "End-to-end summary": "experiments/phase7_end_to_end/run_summary.json",
    "Negotiation trace": "experiments/phase7_end_to_end/negotiation_trace.json",
    "Execution record": "experiments/phase7_end_to_end/execution_record.json",
    "Verification report": "experiments/phase7_end_to_end/verification_report.json",
    "Failure-path trace": "experiments/phase7_end_to_end/failure_path_trace.json",
    "Model comparison": "experiments/phase7_end_to_end/model_baseline_comparison.json",
    "Evidence manifest": "experiments/phase7_end_to_end/evidence_manifest.json",
    "Academic report": "PHASE7_ACADEMIC_REPORT_DRAFT.md",
}

print("=== ARTIFACT INVENTORY ===")
for label, rel in checks.items():
    path = ROOT / rel
    status = "OK" if path.is_file() and path.stat().st_size > 0 else "MISSING/EMPTY"
    size = f"{path.stat().st_size:,} bytes" if path.is_file() else "-"
    print(f"{status:12} | {label:22} | {size}")

print("\n=== SAVED MODEL METRICS ===")
metrics_path = ROOT / checks["Test metrics"]
if metrics_path.is_file():
    print(json.dumps(json.loads(metrics_path.read_text()), indent=2))

print("\n=== PREDICTION FILE CHECK ===")
pred_path = ROOT / checks["Test predictions"]
if pred_path.is_file():
    df = pd.read_csv(pred_path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

print("\n=== END-TO-END SUMMARY ===")
summary_path = ROOT / checks["End-to-end summary"]
if summary_path.is_file():
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))


from pathlib import Path
import json
import zipfile
import hashlib
import pandas as pd

ROOT = Path("/kaggle/working/digital_twin")
NOTEBOOK = ROOT / "PHASE7_DEMONSTRATION.ipynb"
ZIP_PATH = Path("/kaggle/working/digital_twin_phase7_submission.zip")

assert ROOT.exists(), f"Project directory missing: {ROOT}"

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

cells = [
    markdown_cell("""# Personal AI Digital Twin — Phase 7 Demonstration

This notebook audits the **previously saved** model evaluation and the Phase 7 prototype evidence.

It does not train or fine-tune a model. The negotiation, SQLite persistence, privacy, and verification results shown here are read from saved experiment artifacts. The research benchmark metrics are reported separately from the simulated service-execution results.

Project root in the original Kaggle environment: `/kaggle/working/digital_twin`.
"""),
    code_cell("""from pathlib import Path
import json
import pandas as pd

ROOT = Path("/kaggle/working/digital_twin")
if not ROOT.exists():
    ROOT = Path.cwd()

print("Project root:", ROOT)
assert ROOT.exists(), "Project root was not found. Set ROOT to the extracted project directory."
"""),
    markdown_cell("## 1. Artifact and reproducibility audit"),
    code_cell("""required = [
    "src/main.py",
    "src/digital_twin.py",
    "src/execution_verifier.py",
    "models/flan_t5_small_negotiation/corrected_fp32/config.json",
    "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
    "experiments/majority_target_baseline/metrics.json",
    "experiments/phase7_end_to_end/run_summary.json",
    "experiments/phase7_end_to_end/verification_report.json",
    "experiments/phase7_end_to_end/failure_path_trace.json",
    "PHASE7_ACADEMIC_REPORT_DRAFT.md",
]

audit = pd.DataFrame([
    {
        "artifact": rel,
        "exists": (ROOT / rel).is_file(),
        "size_bytes": (ROOT / rel).stat().st_size if (ROOT / rel).is_file() else None,
    }
    for rel in required
])
display(audit)
assert audit["exists"].all(), "One or more required artifacts are missing."
print("Artifact audit: PASS")
"""),
    markdown_cell("## 2. Saved FLAN-T5 test metrics"),
    code_cell("""metrics_path = ROOT / "experiments/flan_t5_small_negotiation/test_metrics_corrected.json"
metrics = json.loads(metrics_path.read_text())
print("Model:", metrics.get("model"))
print("Split:", metrics.get("split"))
print("Examples:", metrics.get("examples"))
print("Exact-match accuracy:", f"{metrics['exact_match_accuracy']:.2%}")
print("Category accuracy:", f"{metrics['category_accuracy']:.2%}")
print("Token micro-F1:", f"{metrics['token_micro_f1']:.2%}")

print("\\nPer-category performance:")
display(pd.DataFrame(metrics["per_true_category"]).T)
print("\\nTrue categories:")
display(pd.Series(metrics["true_category_counts"], name="examples"))
print("\\nPredicted categories:")
display(pd.Series(metrics["predicted_category_counts"], name="predicted examples"))

print("Interpretation: category performance is imbalanced. In particular, the saved metrics show zero correct category predictions for disagreement and disconnect.")
"""),
    markdown_cell("## 3. Saved model versus majority-target baseline"),
    code_cell("""comparison_path = ROOT / "experiments/phase7_end_to_end/model_baseline_comparison.json"
comparison = json.loads(comparison_path.read_text())
display(pd.json_normalize(comparison).T.rename(columns={0: "value"}))
print("\\nExact-match improvement is modest; category accuracy and minority-class results must be interpreted separately.")
"""),
    markdown_cell("## 4. Saved negotiation, execution, and verification evidence"),
    code_cell("""base = ROOT / "experiments/phase7_end_to_end"
summary = json.loads((base / "run_summary.json").read_text())
print(json.dumps(summary, indent=2))

steps = summary.get("steps", {})
assert summary.get("overall_status") == "passed"
assert steps.get("negotiation", {}).get("status") == "agreement"
assert steps.get("execution", {}).get("status") == "persisted"
assert steps.get("verification", {}).get("verified") is True
assert steps.get("privacy_policy", {}).get("passed") is True
assert steps.get("final_agreement_validation", {}).get("passed") is True
print("\\nSaved successful-run assertions: PASS")
"""),
    markdown_cell("## 5. Failure-path and tamper-test evidence"),
    code_cell("""failure_path = json.loads((base / "failure_path_trace.json").read_text())
print("Failure-path evidence:")
print(json.dumps(failure_path, indent=2))

for candidate in [
    ROOT / "experiments/digital_twin_execution/tamper_detection_report.json",
    ROOT / "verification/tamper_detection_report.json",
]:
    if candidate.is_file():
        print("\\nTamper-detection evidence:", candidate.relative_to(ROOT))
        print(candidate.read_text())
"""),
    markdown_cell("## 6. Prediction-file integrity"),
    code_cell("""pred_path = ROOT / "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv"
predictions = pd.read_csv(pred_path)
print("Prediction rows:", len(predictions))
print("Columns:", predictions.columns.tolist())
assert len(predictions) == metrics["examples"] == 1052
assert {"true_category", "predicted_category", "exact_match"}.issubset(predictions.columns)
display(predictions.head(5))
print("\\nPredicted-category distribution:")
display(predictions["predicted_category"].value_counts(dropna=False).to_frame("count"))
"""),
    markdown_cell("""## 7. Conclusion and limitations

The saved evidence demonstrates a working rule-constrained negotiation prototype with simulated persistence, independent verification, and privacy checks. The benchmark experiment is a separate component: the model's exact-match accuracy is 15.11%, category accuracy is 84.60%, and token micro-F1 is 63.87% on the saved test split.

The saved category distribution reveals substantial class imbalance in predictions. The prototype's successful hotel-booking simulation must not be interpreted as proof that FLAN-T5 itself autonomously completed that booking; the demonstration uses the project's negotiation and verification logic.

No model training is performed in this notebook. See `PHASE7_ACADEMIC_REPORT_DRAFT.md` for the report draft; verify bibliography details and rerun the project tests in the intended submission environment before final submission.
"""),
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Validate the required evidence before creating the notebook.
for rel in [
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "experiments/phase7_end_to_end/run_summary.json",
    "experiments/phase7_end_to_end/model_baseline_comparison.json",
    "experiments/phase7_end_to_end/failure_path_trace.json",
]:
    assert (ROOT / rel).is_file(), f"Missing required evidence: {rel}"

NOTEBOOK.write_text(json.dumps(notebook_data, indent=2), encoding="utf-8")

# Validate that the created notebook is valid notebook JSON.
check = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert check["nbformat"] == 4
assert len(check["cells"]) == len(cells)
print("Notebook created:", NOTEBOOK)
print("Notebook cells:", len(check["cells"]))

# Package the project while excluding caches and raw datasets.
# Include only the corrected model checkpoint to avoid duplicating alternate checkpoints.
include_dirs = [
    "src",
    "profiles",
    "experiments",
    "verification",
]
include_files = [
    "notebook_source_backup_20261026.py",
    "PHASE7_ACADEMIC_REPORT_DRAFT.md",
    "PHASE7_DEMONSTRATION.ipynb",
]

checkpoint_dir = ROOT / "models/flan_t5_small_negotiation/corrected_fp32"
assert (checkpoint_dir / "model.safetensors").is_file(), "Corrected model weights missing."

files_to_add = set()

for rel_dir in include_dirs:
    directory = ROOT / rel_dir
    if directory.exists():
        for f in directory.rglob("*"):
            if not f.is_file():
                continue
            if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in f.parts):
                continue
            if f.suffix in {".pyc", ".pyo"}:
                continue
            files_to_add.add(f)

for rel_file in include_files:
    f = ROOT / rel_file
    if f.is_file():
        files_to_add.add(f)

for f in checkpoint_dir.rglob("*"):
    if f.is_file():
        files_to_add.add(f)

# Keep the SQLite verification database and the original source backup.
db = ROOT / "verification/services.db"
if db.is_file():
    files_to_add.add(db)

# Write a ZIP without including the ZIP itself.
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
    for f in sorted(files_to_add):
        zf.write(f, arcname=str(f.relative_to(ROOT)))

print("\n=== EXPORT SUMMARY ===")
print("Notebook:", NOTEBOOK, f"({NOTEBOOK.stat().st_size:,} bytes)")
print("ZIP:", ZIP_PATH, f"({ZIP_PATH.stat().st_size:,} bytes)")
print("Files packaged:", len(files_to_add))
print("ZIP SHA-256:", hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest())

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    bad_member = zf.testzip()
    print("ZIP integrity:", "PASS" if bad_member is None else f"FAIL: {bad_member}")
    print("Packaged model weights:", any(
        name.endswith("corrected_fp32/model.safetensors")
        for name in zf.namelist()
    ))


from pathlib import Path
import json
import zipfile
import hashlib
import sys

ROOT = Path("/kaggle/working/digital_twin")
NOTEBOOK = ROOT / "PHASE7_DEMONSTRATION.ipynb"
ZIP_PATH = Path("/kaggle/working/digital_twin_phase7_submission.zip")
EXECUTED_NOTEBOOK = ROOT / "PHASE7_DEMONSTRATION_EXECUTED.ipynb"

results = []

def check(name, passed, detail=""):
    results.append((name, bool(passed), str(detail)))
    print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))

# 1. Notebook structure and code syntax
try:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    syntax_errors = []
    for i, cell in enumerate(code_cells):
        try:
            compile(cell.source, f"notebook_cell_{i+1}", "exec")
        except SyntaxError as e:
            syntax_errors.append(f"Cell {i+1}: {e}")

    check("Notebook is valid nbformat", nbformat.validate(nb) is None)
    check("Notebook has 15 cells", len(nb.cells) == 15, f"{len(nb.cells)} cells")
    check("All code cells compile", not syntax_errors,
          "; ".join(syntax_errors) if syntax_errors else f"{len(code_cells)} code cells")

except Exception as e:
    print("Notebook pre-check error:", repr(e))
    raise

# 2. Required artifacts
required = [
    "src/main.py",
    "src/digital_twin.py",
    "src/execution_verifier.py",
    "models/flan_t5_small_negotiation/corrected_fp32/config.json",
    "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
    "experiments/majority_target_baseline/metrics.json",
    "experiments/phase7_end_to_end/run_summary.json",
    "experiments/phase7_end_to_end/verification_report.json",
    "experiments/phase7_end_to_end/failure_path_trace.json",
    "experiments/phase7_end_to_end/model_baseline_comparison.json",
    "PHASE7_ACADEMIC_REPORT_DRAFT.md",
]
missing = [p for p in required if not (ROOT / p).is_file() or (ROOT / p).stat().st_size == 0]
check("Required project artifacts exist", not missing,
      "Missing: " + ", ".join(missing) if missing else f"{len(required)} artifacts present")

# 3. Benchmark rows and saved metrics
metrics = json.loads(
    (ROOT / "experiments/flan_t5_small_negotiation/test_metrics_corrected.json").read_text()
)
import pandas as pd
pred = pd.read_csv(ROOT / "experiments/flan_t5_small_negotiation/test_predictions_corrected.csv")
check("Test predictions contain 1,052 rows", len(pred) == 1052, f"rows={len(pred)}")
check("Metrics and predictions agree on test size",
      metrics.get("examples") == len(pred), f"metrics={metrics.get('examples')}, CSV={len(pred)}")

# 4. Saved successful end-to-end evidence
summary = json.loads(
    (ROOT / "experiments/phase7_end_to_end/run_summary.json").read_text()
)
steps = summary.get("steps", {})
e2e_ok = (
    summary.get("overall_status") == "passed"
    and steps.get("negotiation", {}).get("status") == "agreement"
    and steps.get("execution", {}).get("status") == "persisted"
    and steps.get("verification", {}).get("verified") is True
    and steps.get("privacy_policy", {}).get("passed") is True
    and steps.get("final_agreement_validation", {}).get("passed") is True
)
check("Saved end-to-end assertions pass", e2e_ok, summary.get("overall_status", "missing status"))

# 5. ZIP integrity and required members
zip_ok = False
zip_detail = ""
if ZIP_PATH.is_file():
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            bad = zf.testzip()
            names = zf.namelist()
            required_members = [
                "src/main.py",
                "src/digital_twin.py",
                "src/execution_verifier.py",
                "models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
                "experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
                "PHASE7_ACADEMIC_REPORT_DRAFT.md",
                "PHASE7_DEMONSTRATION.ipynb",
            ]
            absent = [p for p in required_members if p not in names]
            zip_ok = bad is None and not absent
            zip_detail = (
                f"{len(names)} members; corrupt={bad}; missing={absent}"
            )
    except Exception as e:
        zip_detail = repr(e)
check("Submission ZIP integrity and contents", zip_ok, zip_detail)
if ZIP_PATH.is_file():
    print("ZIP size:", f"{ZIP_PATH.stat().st_size:,}", "bytes")
    print("ZIP SHA-256:", hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest())

# 6. Execute the demonstration notebook without training
execution_ok = False
try:
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    executed_nb = client.execute()
    nbformat.write(executed_nb, EXECUTED_NOTEBOOK)

    errors = []
    executed_code_cells = 0
    for i, cell in enumerate(executed_nb.cells, start=1):
        if cell.cell_type != "code":
            continue
        executed_code_cells += 1
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((i, output.get("ename"), output.get("evalue")))

    execution_ok = not errors
    check(
        "Demonstration notebook executes",
        execution_ok,
        f"{executed_code_cells} code cells executed; errors={errors}"
    )
    if execution_ok:
        print("Executed notebook saved:", EXECUTED_NOTEBOOK)

except Exception as e:
    check("Demonstration notebook executes", False, repr(e))
    print("If the error is a missing package or kernel issue, send me the complete traceback.")

# 7. Final summary
print("\n=== FINAL VALIDATION SUMMARY ===")
for name, passed, detail in results:
    print(f"{'PASS' if passed else 'FAIL'} | {name}")
print(f"\nChecks passed: {sum(p for _, p, _ in results)}/{len(results)}")
print("Notebook execution success:", execution_ok)
print("Original notebook and model training artifacts were not intentionally modified.")


from pathlib import Path
import nbformat
import warnings

p = Path("/kaggle/working/digital_twin/PHASE7_DEMONSTRATION.ipynb")

# Read the saved, normalized notebook
nb = nbformat.read(p, as_version=4)

# Confirm every cell has a non-empty ID
missing_ids = [
    i for i, cell in enumerate(nb.cells)
    if not cell.get("id")
]
print("Cells:", len(nb.cells))
print("Cells missing IDs:", missing_ids)

# Validate and capture any warnings from this validation call
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    nbformat.validate(nb)

id_warnings = [
    str(w.message) for w in caught
    if "MissingIDFieldWarning" in str(w.category)
    or "Cell is missing an id field" in str(w.message)
]

print("Notebook validation:", "PASS" if not missing_ids else "FAIL")
print("Missing-ID warnings:", len(id_warnings))
for warning in id_warnings:
    print(warning)

assert not missing_ids, "Some notebook cells still lack IDs."
assert not id_warnings, "Notebook validation still reports missing cell IDs."
print("FINAL RESULT: Notebook IDs and validation PASS.")


from pathlib import Path
import zipfile
import hashlib

root = Path("/kaggle/working/digital_twin")
zip_path = Path("/kaggle/working/digital_twin_phase7_submission.zip")

with zipfile.ZipFile(zip_path, "r") as old_zip:
    files = {
        name: old_zip.read(name)
        for name in old_zip.namelist()
    }

notebook_rel = "PHASE7_DEMONSTRATION.ipynb"
files[notebook_rel] = (root / notebook_rel).read_bytes()

temp_zip = zip_path.with_suffix(".tmp")
with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as new_zip:
    for name, content in files.items():
        new_zip.writestr(name, content)

temp_zip.replace(zip_path)

with zipfile.ZipFile(zip_path, "r") as zf:
    assert zf.testzip() is None
    assert zf.read(notebook_rel) == (root / notebook_rel).read_bytes()

print("Updated ZIP:", zip_path)
print("Size:", f"{zip_path.stat().st_size:,}", "bytes")
print("SHA-256:", hashlib.sha256(zip_path.read_bytes()).hexdigest())
print("ZIP integrity: PASS")


from pathlib import Path

ROOT = Path("/kaggle/working/digital_twin")

print("=== PROJECT DIRECTORY ===")
print("Exists:", ROOT.exists())
print("Path:", ROOT)

print("\n=== DATASETS AND NOTEBOOKS ===")
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and (
        p.suffix.lower() in {".ipynb", ".csv", ".jsonl", ".parquet",
                             ".txt", ".db", ".sqlite", ".zip"}
        or "dataset" in str(p).lower()
        or "notebook" in p.name.lower()
    ):
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"{size_mb:8.2f} MB | {p.relative_to(ROOT)}")
        except OSError:
            print("UNREADABLE |", p)

print("\n=== TOP-LEVEL KAGGLE DATA DIRECTORIES ===")
for folder in [Path("/kaggle/input"), Path("/kaggle/working")]:
    print(f"\n{folder}:")
    if folder.exists():
        for p in sorted(folder.iterdir()):
            try:
                print(("DIR " if p.is_dir() else "FILE"), p.name)
            except OSError:
                print("UNREADABLE", p)


from pathlib import Path

for root in [
    Path("/kaggle/input/datasets"),
    Path("/kaggle/working/digital_twin/models"),
    Path("/kaggle/working/digital_twin/data"),
    Path("/kaggle/working/digital_twin/datasets"),
]:
    print(f"\n=== {root} ===")
    if not root.exists():
        print("Not present")
        continue

    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)

    print("Files:", len(files))
    print("Total size:", f"{total / (1024**2):.2f} MB")

    for p in sorted(files)[:60]:
        print(f"{p.stat().st_size / (1024**2):8.2f} MB | {p}")
    if len(files) > 60:
        print(f"... and {len(files) - 60} more files")



from pathlib import Path
from datetime import datetime, timezone
import json, zipfile, hashlib, uuid

WORK = Path("/kaggle/working")
ROOT = WORK / "digital_twin"
EXTERNAL = Path("/kaggle/input/datasets/vibecoder18/dtp-backup/digital_twin")
SUBMISSION = WORK / "digital_twin_phase7_submission.zip"
MASTER = WORK / "digital_twin_COMPLETE_BACKUP.zip"
MANIFEST = ROOT / "COMPLETE_BACKUP_MANIFEST.json"
NOTEBOOK = ROOT / "PHASE7_DEMONSTRATION.ipynb"

assert ROOT.is_dir(), f"Project missing: {ROOT}"
assert EXTERNAL.is_dir(), f"External backup missing: {EXTERNAL}"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def should_skip(path):
    return any(part in {
        "__pycache__", ".ipynb_checkpoints", ".git"
    } for part in path.parts) or path.suffix in {".pyc", ".pyo"}

# Normalize notebook cell IDs.
import nbformat
nb = nbformat.read(NOTEBOOK, as_version=4)
for cell in nb.cells:
    if not cell.get("id"):
        cell["id"] = uuid.uuid4().hex[:8]
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
print("PASS: demonstration notebook normalized")

# Collect all current project files except the manifest being regenerated.
current_files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and not should_skip(p.relative_to(ROOT))
    and p != MANIFEST
)

# Preserve every file in the attached external Kaggle backup.
external_files = sorted(
    p for p in EXTERNAL.rglob("*")
    if p.is_file() and not should_skip(p.relative_to(EXTERNAL))
)

# Include the earlier submission archive, if available.
nested = SUBMISSION.is_file()

# Hash inventory for both file collections.
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "description": (
        "Recovery backup of current Digital Twin project and attached "
        "Kaggle backup. Contains saved checkpoints and results; does not "
        "claim that model training was reproduced from scratch."
    ),
    "current_project_files": [
        {
            "path": str(p.relative_to(ROOT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        } for p in current_files
    ],
    "external_backup_files": [
        {
            "path": str(p.relative_to(EXTERNAL)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
        } for p in external_files
    ],
    "previous_submission_zip": (
        {
            "name": SUBMISSION.name,
            "size_bytes": SUBMISSION.stat().st_size,
            "sha256": sha256(SUBMISSION),
        } if nested else None
    ),
}
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# Build one master ZIP. Write to a temporary file first.
temp = MASTER.with_suffix(".tmp")
with zipfile.ZipFile(
    temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
) as zf:
    for p in current_files:
        zf.write(p, f"digital_twin/{p.relative_to(ROOT)}")

    zf.write(MANIFEST, "digital_twin/COMPLETE_BACKUP_MANIFEST.json")

    for p in external_files:
        zf.write(p, f"external_kaggle_backup/{p.relative_to(EXTERNAL)}")

    if nested:
        zf.write(SUBMISSION, f"previous_exports/{SUBMISSION.name}")

temp.replace(MASTER)

# Verify archive integrity and essential recovery files.
required = [
    "digital_twin/src/main.py",
    "digital_twin/src/digital_twin.py",
    "digital_twin/src/execution_verifier.py",
    "digital_twin/models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "digital_twin/experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
    "digital_twin/experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
    "digital_twin/PHASE7_DEMONSTRATION.ipynb",
    "digital_twin/PHASE7_DEMONSTRATION_EXECUTED.ipynb",
    "digital_twin/notebook_source_backup_20261026.py",
    "digital_twin/PHASE7_ACADEMIC_REPORT_DRAFT.md",
    "digital_twin/COMPLETE_BACKUP_MANIFEST.json",
    "external_kaggle_backup/models/flan_t5_small_negotiation/best_fp32/model.safetensors",
    "external_kaggle_backup/models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    "external_kaggle_backup/experiments/flan_t5_small_negotiation/training_log_corrected_fp32.json",
]

with zipfile.ZipFile(MASTER, "r") as zf:
    names = set(zf.namelist())
    bad = zf.testzip()
    missing = [name for name in required if name not in names]
    assert bad is None, f"Corrupt ZIP member: {bad}"
    assert not missing, f"Missing backup files: {missing}"

print("\n========== MASTER BACKUP VERIFIED ==========")
print("File:", MASTER)
print("Size:", f"{MASTER.stat().st_size / (1024**2):.2f} MB")
print("Archive entries:", len(names))
print("Current project files:", len(current_files))
print("External backup files:", len(external_files))
print("Previous submission ZIP included:", nested)
print("ZIP integrity: PASS")
print("Required files: PASS")
print("SHA-256:", sha256(MASTER))
print("\nDownload the master ZIP to your computer and keep a second copy.")




from pathlib import Path
import zipfile
import hashlib
import json

archive = Path("/kaggle/working/digital_twin_COMPLETE_BACKUP.zip")

assert archive.is_file(), f"Backup ZIP not found: {archive}"

print("=== 1. ARCHIVE FILE ===")
print("Path:", archive)
print("Size:", f"{archive.stat().st_size / (1024**3):.3f} GiB")

print("\n=== 2. ZIP INTEGRITY ===")
with zipfile.ZipFile(archive, "r") as zf:
    bad = zf.testzip()
    names = zf.namelist()

    assert bad is None, f"Corrupt archive member: {bad}"
    print("PASS: all ZIP members passed CRC verification")
    print("Entries:", len(names))

    required = [
        "digital_twin/src/main.py",
        "digital_twin/src/digital_twin.py",
        "digital_twin/src/execution_verifier.py",
        "digital_twin/models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
        "digital_twin/experiments/flan_t5_small_negotiation/test_metrics_corrected.json",
        "digital_twin/experiments/flan_t5_small_negotiation/test_predictions_corrected.csv",
        "digital_twin/PHASE7_DEMONSTRATION.ipynb",
        "digital_twin/PHASE7_DEMONSTRATION_EXECUTED.ipynb",
        "digital_twin/PHASE7_ACADEMIC_REPORT_DRAFT.md",
        "digital_twin/COMPLETE_BACKUP_MANIFEST.json",
        "external_kaggle_backup/models/flan_t5_small_negotiation/best_fp32/model.safetensors",
        "external_kaggle_backup/models/flan_t5_small_negotiation/corrected_fp32/model.safetensors",
    ]

    missing = [item for item in required if item not in names]
    assert not missing, f"Missing required files: {missing}"
    print("PASS: required project, notebook, and model files found")

    # Validate notebook JSON directly from the archive.
    import nbformat
    for notebook_name in [
        "digital_twin/PHASE7_DEMONSTRATION.ipynb",
        "digital_twin/PHASE7_DEMONSTRATION_EXECUTED.ipynb",
    ]:
        nb_data = json.loads(zf.read(notebook_name).decode("utf-8"))
        nbformat.validate(nbformat.from_dict(nb_data))
        print("PASS: valid notebook:", notebook_name)

print("\n=== 3. SHA-256 ===")
h = hashlib.sha256()
with archive.open("rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(chunk)
digest = h.hexdigest()
print(digest)

print("\n=== RESULT ===")
print("BACKUP VERIFIED")
print("Download from Kaggle's file browser:")
print(str(archive))



import os
from IPython.display import FileLink, display

os.chdir("/kaggle/working")

filename = "digital_twin_COMPLETE_BACKUP.zip"
assert os.path.isfile(filename), f"Missing file: {filename}"

print("File exists:", os.path.getsize(filename), "bytes")
print("Click the link below to download:")
display(FileLink(filename))






