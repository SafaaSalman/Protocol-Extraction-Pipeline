"""
Test protocol extraction on the 3 held-out gold standard protocols.

Few-shot examples: CPR, Burn Injuries (included in prompt)
Evaluation targets: Patient Assessment, Risk Management, Multi-Casualty Triage
"""

import sys
import json

sys.path.insert(0, ".")

from src.ingestion.pdf_ingest import ingest_pdf
from src.extraction.section_detector import detect_sections
from src.extraction.protocol_extractor import extract_protocol, save_extraction

# Ingest and detect
print("Ingesting PDF...")
doc = ingest_pdf("IRPG-2025.pdf")
print("Detecting sections...")
result = detect_sections(doc)

# Find the 3 held-out protocols by title
targets = [
    "Patient Assessment",
    "Risk Management Process",
    "Multi-Casualty Triage System",
]

target_sections = []
for section in result.sections:
    if section.title in targets:
        target_sections.append(section)

print(f"\nFound {len(target_sections)} target sections:")
for s in target_sections:
    print(f"  {s.title} (p.{s.start_page}-{s.end_page}, {s.format_hint})")

# Extract each
for section in target_sections:
    print(f"\n{'='*60}")
    print(f"Extracting: {section.title}")
    print(f"{'='*60}")

    extracted = extract_protocol(section, doc, model="gpt-4o", use_few_shot=True)

    # Save
    slug = section.title.lower().replace(" ", "_").replace("-", "_")
    out_path = f"evaluation/extracted/{slug}.json"
    save_extraction(extracted, out_path)

    # Summary
    nodes = len(extracted.get("nodes", []))
    edges = len(extracted.get("edges", []))
    ptype = extracted.get("protocol_type", "?")
    conf = extracted.get("extraction_confidence", 0)
    meta = extracted.get("_extraction_meta", {})
    tokens = meta.get("total_tokens", 0)

    print(f"  Type: {ptype}, Confidence: {conf}")
    print(f"  Nodes: {nodes}, Edges: {edges}")
    print(f"  Tokens used: {tokens}")
    print(f"  Saved to: {out_path}")

print("\nDone!")
