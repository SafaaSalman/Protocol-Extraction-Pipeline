"""
Targeted test of the batch extractor:
1. Pre-classify a sample of borderline VIC sections (gpt-4o-mini)
2. Extract 3 VIC protocols with TOC context (gpt-4o)
"""

import sys
import json
import time

sys.path.insert(0, ".")

from openai import OpenAI
from src.ingestion.pdf_ingest import ingest_pdf
from src.extraction.section_detector import detect_sections
from src.extraction.batch_extractor import (
    BatchConfig,
    plan_batch,
    build_toc_context,
    pre_classify_section,
)
from src.extraction.protocol_extractor import (
    extract_protocol,
    extract_protocol_multimodal,
    save_extraction,
)

# ── Ingest VIC PDF ───────────────────────────────────────────────────
print("Ingesting VIC-BushfireHandbook.pdf...")
doc = ingest_pdf("VIC-BushfireHandbook.pdf")
detection = detect_sections(doc)
config = BatchConfig()

to_extract, to_classify, to_skip = plan_batch(detection, config)
print(f"Extract: {len(to_extract)}, Classify: {len(to_classify)}, Skip: {len(to_skip)}\n")

client = OpenAI()

# ── Part 1: Pre-classify 10 diverse borderline sections ─────────────
print("=" * 60)
print("PART 1: Pre-classification (gpt-4o-mini)")
print("=" * 60)

# Pick a diverse sample: some that should be extractable, some not
sample_titles = [
    "Definitions",                    # Probably NOT extractable
    "Incident Action Planning",       # Has table, might be extractable
    "Transfer of Control",            # Could be a process
    "Staging Areas",                  # Probably descriptive
    "LACES",                          # Safety acronym - has steps
    "WATCHOUT",                       # Safety acronym
    "Red Flag Warnings",              # Might have a process
    "Fire Danger Ratings",            # Table-based, probably not a process
    "Burn Injuries",                  # IRPG gold standard - should be extractable
    "Community Meetings",             # Probably descriptive
]

sample_sections = []
for title in sample_titles:
    matches = [s for s in to_classify if title in s.title]
    if matches:
        sample_sections.append(matches[0])

print(f"Testing {len(sample_sections)} borderline sections:\n")

for section in sample_sections:
    start = time.time()
    is_extractable = pre_classify_section(section, doc, client)
    elapsed = time.time() - start
    verdict = "EXTRACT" if is_extractable else "SKIP"
    print(f"  [{verdict}] {section.title} "
          f"(p.{section.start_page}-{section.end_page}, "
          f"conf={section.confidence:.2f}, {section.format_hint}) "
          f"[{elapsed:.1f}s]")


# ── Part 2: Extract 3 VIC protocols ─────────────────────────────────
print(f"\n{'=' * 60}")
print("PART 2: Extract 3 VIC protocols (gpt-4o)")
print("=" * 60)

# Pick 3 diverse protocols from the extract tier
extract_targets = [
    ("Evacuation", False),                # Sequential, 4 pages, tables
    ("Dynamic Risk Assessment", True),    # Decision tree, likely has diagrams
    ("Traffic Management Points", True),  # Hybrid, tables + decisions
]

for title_substr, use_multimodal in extract_targets:
    section = next(s for s in to_extract if title_substr in s.title)
    toc_ctx = build_toc_context(section, detection)

    print(f"\n  Extracting: {section.title}")
    print(f"  Pages: {section.start_page}-{section.end_page}")
    print(f"  Format: {section.format_hint}, conf={section.confidence:.2f}")
    print(f"  Multimodal: {use_multimodal}")
    print(f"  Context: {toc_ctx}")

    start = time.time()

    if use_multimodal:
        graph = extract_protocol_multimodal(
            section, doc, "VIC-BushfireHandbook.pdf",
            model="gpt-4o",
            use_few_shot=True,
            toc_context=toc_ctx,
        )
    else:
        graph = extract_protocol(
            section, doc,
            model="gpt-4o",
            use_few_shot=True,
            toc_context=toc_ctx,
        )

    elapsed = time.time() - start
    meta = graph.get("_extraction_meta", {})
    tokens = meta.get("total_tokens", 0)

    n_nodes = len(graph.get("nodes", []))
    n_edges = len(graph.get("edges", []))
    node_types = {}
    for n in graph.get("nodes", []):
        t = n.get("type", "?")
        node_types[t] = node_types.get(t, 0) + 1

    print(f"  → {n_nodes} nodes, {n_edges} edges, "
          f"types={node_types}, {tokens} tokens, {elapsed:.1f}s")
    print(f"  → protocol_type={graph.get('protocol_type')}, "
          f"confidence={graph.get('extraction_confidence')}")

    # Save
    safe_name = section.title.lower().replace(" ", "_").replace("-", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    out_path = f"evaluation/extracted/vic_{safe_name}.json"
    save_extraction(graph, out_path)
    print(f"  → Saved to {out_path}")

print("\nDone!")
