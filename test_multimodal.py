"""
Re-extract triage protocol using multimodal (text + page image) extraction
and compare with both the text-only extraction and the gold standard.
"""

import sys
import json

sys.path.insert(0, ".")

from src.ingestion.pdf_ingest import ingest_pdf
from src.extraction.section_detector import detect_sections
from src.extraction.protocol_extractor import extract_protocol_multimodal, save_extraction

PDF_PATH = "IRPG-2025.pdf"

print("Ingesting PDF...")
doc = ingest_pdf(PDF_PATH)
print("Detecting sections...")
result = detect_sections(doc)

# Find triage section
triage = None
for section in result.sections:
    if section.title == "Multi-Casualty Triage System":
        triage = section
        break

if not triage:
    print("ERROR: Triage section not found")
    sys.exit(1)

print(f"Found: {triage.title} (p.{triage.start_page}-{triage.end_page}, {triage.format_hint})")

print(f"\nExtracting with multimodal (text + image)...")
extracted = extract_protocol_multimodal(
    triage, doc, PDF_PATH, model="gpt-4o", use_few_shot=True, dpi=200
)

out_path = "evaluation/extracted/triage_multimodal.json"
save_extraction(extracted, out_path)

nodes = len(extracted.get("nodes", []))
edges = len(extracted.get("edges", []))
meta = extracted.get("_extraction_meta", {})
tokens = meta.get("total_tokens", 0)

print(f"  Nodes: {nodes}, Edges: {edges}")
print(f"  Tokens used: {tokens}")
print(f"  Saved to: {out_path}")

# Quick comparison
print(f"\n{'='*60}")
print("Quick comparison: Gold vs Text-only vs Multimodal")
print(f"{'='*60}")

with open("evaluation/gold/triage_01.json") as f:
    gold = json.load(f)
with open("evaluation/extracted/multi_casualty_triage_system.json") as f:
    text_only = json.load(f)

for label, data in [("Gold standard", gold), ("Text-only", text_only), ("Multimodal", extracted)]:
    n = len(data.get("nodes", []))
    e = len(data.get("edges", []))
    types = {}
    for node in data.get("nodes", []):
        t = node["type"]
        types[t] = types.get(t, 0) + 1
    print(f"\n  {label}:")
    print(f"    Nodes: {n}  Edges: {e}")
    print(f"    Node types: {types}")

print("\nDone! Run evaluate.py with the multimodal file to get full metrics.")
