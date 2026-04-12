"""Quick structural validation test on freshly extracted protocols."""
import sys
import json

sys.path.insert(0, ".")

from src.extraction.validator import validate_structure

files = [
    "evaluation/extracted/risk_management_process.json",
    "evaluation/extracted/patient_assessment.json",
    "evaluation/extracted/multi_casualty_triage_system.json",
]

for filepath in files:
    with open(filepath) as fh:
        data = json.load(fh)
    issues = validate_structure(data)
    name = filepath.split("/")[-1]
    print(f"\n{name}:")
    print(f"  Valid: {len(issues) == 0}")
    if issues:
        for issue in issues:
            print(f"  Issue: {issue}")
    else:
        print("  All structural checks passed")
