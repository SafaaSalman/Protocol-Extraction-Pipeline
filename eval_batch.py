"""Compare batch-extracted (with validator) vs gold standards."""
import json
import sys
import os
sys.path.insert(0, ".")
from evaluate import full_evaluation

pairs = [
    ("evaluation/gold/risk_management_01.json", "evaluation/extracted/risk_management_process.json", "Risk Mgmt (batch+val)"),
    ("evaluation/gold/patient_assessment_01.json", "evaluation/extracted/patient_assessment.json", "Patient Assess (batch+val)"),
    ("evaluation/gold/triage_01.json", "evaluation/extracted/multicasualty_triage_system.json", "Triage (batch+val)"),
    ("evaluation/gold/cpr_01.json", "evaluation/extracted/cpr.json", "CPR (batch+val)"),
    ("evaluation/gold/burn_injuries_01.json", "evaluation/extracted/burn_injuries.json", "Burn Injuries (batch+val)"),
]

# Previous results for comparison (from before batch pipeline)
prev = {
    "Risk Mgmt": {"node_f1": 0.703, "edge_f1": 0.488},
    "Patient Assess": {"node_f1": 0.757, "edge_f1": 0.414},
    "Triage (text)": {"node_f1": 0.667, "edge_f1": 0.083},
    "Triage (mm)": {"node_f1": 0.640, "edge_f1": 0.444},
}

print("=" * 100)
print("EVALUATION: Batch+Validator vs Gold Standards")
print("=" * 100)
print()

header = "{:28s} | {:>6s} {:>6s} {:>6s} | {:>6s} {:>6s} {:>6s} | {:>5s} | Sizes".format(
    "Protocol", "NP", "NR", "NF1", "EP", "ER", "EF1", "Valid"
)
print(header)
print("-" * 100)

all_nf1 = []
all_ef1 = []

for gold_path, ext_path, label in pairs:
    if not os.path.exists(gold_path) or not os.path.exists(ext_path):
        print(label + ": FILES MISSING")
        continue

    with open(gold_path) as f:
        gold = json.load(f)
    with open(ext_path) as f:
        ext = json.load(f)

    r = full_evaluation(gold, ext)
    n = r["nodes"]
    e = r["edges"]
    s = r["structure"]

    ng = len(gold.get("nodes", []))
    ne = len(ext.get("nodes", []))
    eg = len(gold.get("edges", []))
    ee = len(ext.get("edges", []))

    validated = ext.get("_validated", False)
    tag = " [V]" if validated else ""

    line = "{:28s} | {:6.3f} {:6.3f} {:6.3f} | {:6.3f} {:6.3f} {:6.3f} | {:>5s} | {}g/{}e nodes, {}g/{}e edges{}".format(
        label,
        n["precision"], n["recall"], n["f1"],
        e["precision"], e["recall"], e["f1"],
        str(s["valid"]),
        ng, ne, eg, ee, tag
    )
    print(line)

    if s["issues"]:
        for issue in s["issues"]:
            print("  ISSUE: " + issue)

    all_nf1.append(n["f1"])
    all_ef1.append(e["f1"])

print("-" * 100)
if all_nf1:
    avg_nf1 = sum(all_nf1) / len(all_nf1)
    avg_ef1 = sum(all_ef1) / len(all_ef1)
    print("{:28s} | {:>6s} {:>6s} {:6.3f} | {:>6s} {:>6s} {:6.3f}".format(
        "AVERAGE", "", "", avg_nf1, "", "", avg_ef1
    ))

print()
print("Previous results (before batch pipeline):")
print("  Risk Mgmt:       Node F1=0.703, Edge F1=0.488")
print("  Patient Assess:  Node F1=0.757, Edge F1=0.414")
print("  Triage (text):   Node F1=0.667, Edge F1=0.083")
print("  Triage (mm):     Node F1=0.640, Edge F1=0.444")
print("  Avg:             Node F1=0.692, Edge F1=0.357")
