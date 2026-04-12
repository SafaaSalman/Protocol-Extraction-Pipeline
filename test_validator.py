"""
Test the validator on the 3 VIC protocols we just extracted.
"""
import sys
import json
import time

sys.path.insert(0, ".")

from src.ingestion.pdf_ingest import ingest_pdf
from src.extraction.section_detector import detect_sections
from src.extraction.validator import (
    validate_structure,
    validate_protocol,
    print_validation_summary,
)

# Ingest VIC PDF to get source text
print("Ingesting VIC-BushfireHandbook.pdf...")
doc = ingest_pdf("VIC-BushfireHandbook.pdf")
detection = detect_sections(doc)

# Map sections by title for lookup
section_map = {s.title: s for s in detection.sections}


def get_section_text(section):
    parts = []
    for pg_num in range(section.start_page, section.end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        page = doc.pages[pg_num - 1]
        parts.append(f"[Page {pg_num}]")
        parts.append(page.full_text)
        for table in page.tables:
            parts.append(f"\n[Table on page {pg_num}]")
            for row in table.cells:
                parts.append(" | ".join(cell for cell in row))
    return "\n".join(parts)


# Test cases
test_files = [
    ("vic_evacuation.json", "Evacuation"),
    ("vic_dynamic_risk_assessment.json", "Dynamic Risk Assessment"),
    ("vic_traffic_management_points.json", "Traffic Management Points"),
]

for filename, section_title in test_files:
    print("\n" + "=" * 60)
    print(f"VALIDATING: {section_title}")
    print("=" * 60)

    # Load extracted graph
    with open(f"evaluation/extracted/{filename}") as f:
        graph = json.load(f)

    # Get source text
    section = section_map[section_title]
    section_text = get_section_text(section)

    # Step 1: Structural checks
    structural = validate_structure(graph)
    print(f"\nStructural issues: {len(structural)}")
    for issue in structural:
        marker = "ERR" if issue.severity == "error" else "WARN"
        print(f"  [{marker}] {issue.message}")

    # Step 2: Full validation (structural + semantic)
    print("\nRunning semantic validation (gpt-4o)...")
    start = time.time()
    result = validate_protocol(graph, section_text, run_semantic=True)
    elapsed = time.time() - start

    print_validation_summary(result)
    print(f"Time: {elapsed:.1f}s")

    if result.was_corrected:
        # Show diff summary
        orig_nodes = len(graph.get("nodes", []))
        orig_edges = len(graph.get("edges", []))
        corr = result.corrected_graph
        corr_nodes = len(corr.get("nodes", []))
        corr_edges = len(corr.get("edges", []))
        print(f"Before: {orig_nodes} nodes, {orig_edges} edges")
        print(f"After:  {corr_nodes} nodes, {corr_edges} edges")

        # Save corrected version
        from src.extraction.protocol_extractor import save_extraction
        corr_path = f"evaluation/extracted/{filename.replace('.json', '_validated.json')}"
        save_extraction(corr, corr_path)
        print(f"Saved corrected graph to {corr_path}")

print("\nDone!")
