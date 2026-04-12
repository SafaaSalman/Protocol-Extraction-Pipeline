"""Generate evaluation/manifest.json for the viewer."""
import json, os

GOLD_DIR = "evaluation/gold"
EXT_DIR = "evaluation/extracted"

SOURCE_SHORT = {
    "IRPG-2025.pdf": "IRPG",
    "VIC-BushfireHandbook.pdf": "VIC",
}

# Old test files to skip (superseded by batch run)
SKIP = {
    "multi_casualty_triage_system.json",  # old pre-batch
    "triage_multimodal.json",             # old multimodal test
    "vic_dynamic_risk_assessment.json",   # old VIC test
    "vic_dynamic_risk_assessment_validated.json",
    "vic_evacuation.json",
    "vic_traffic_management_points.json",
}

manifest = {"gold": [], "extracted": []}

# Gold standards
for f in sorted(os.listdir(GOLD_DIR)):
    if not f.endswith(".json"):
        continue
    path = f"{GOLD_DIR}/{f}"
    with open(path) as fh:
        data = json.load(fh)
    manifest["gold"].append({
        "file": f,
        "path": path,
        "title": data.get("title", f.replace(".json", "")),
        "nodes": len(data.get("nodes", [])),
        "edges": len(data.get("edges", [])),
    })

# Extracted
for f in sorted(os.listdir(EXT_DIR)):
    if not f.endswith(".json") or f in SKIP:
        continue
    path = f"{EXT_DIR}/{f}"
    with open(path) as fh:
        data = json.load(fh)
    source_raw = data.get("source_pdf", "")
    source = SOURCE_SHORT.get(source_raw, source_raw)
    manifest["extracted"].append({
        "file": f,
        "path": path,
        "title": data.get("title", f.replace(".json", "")),
        "source": source,
        "nodes": len(data.get("nodes", [])),
        "edges": len(data.get("edges", [])),
        "validated": data.get("_validated", False),
    })

out = "evaluation/manifest.json"
with open(out, "w") as fh:
    json.dump(manifest, fh, indent=2)

gold_n = len(manifest["gold"])
irpg_n = sum(1 for e in manifest["extracted"] if e["source"] == "IRPG")
vic_n = sum(1 for e in manifest["extracted"] if e["source"] == "VIC")
print(f"Manifest: {gold_n} gold, {irpg_n} IRPG, {vic_n} VIC -> {out}")
