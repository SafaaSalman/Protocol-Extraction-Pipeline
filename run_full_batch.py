"""
Full batch extraction run on both PDFs.
Usage: python run_full_batch.py [vic|irpg|both]
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.extraction.batch_extractor import BatchConfig, BatchResult, run_batch


def run_and_summarize(pdf_name: str, config: BatchConfig) -> BatchResult:
    """Run batch extraction and print detailed summary."""
    print("\n" + "=" * 70)
    print(f"  FULL BATCH: {pdf_name}")
    print("=" * 70 + "\n")

    start = time.time()
    result = run_batch(pdf_name, config)
    total = time.time() - start

    # Detailed summary
    print(f"\n{'─' * 50}")
    print(f"SUMMARY for {pdf_name}")
    print(f"{'─' * 50}")
    print(f"Total sections detected:  {result.total_sections}")
    print(f"Total protocols found:    {result.total_protocols}")
    print(f"Extracted:                {result.extracted}")
    print(f"Skipped (low confidence): {result.skipped_low_conf}")
    print(f"Skipped (classifier):     {result.skipped_classify}")
    print(f"Errors:                   {result.errors}")
    print(f"Total tokens:             {result.total_tokens:,}")
    print(f"Total time:               {total:.1f}s")

    # Per-extraction details
    extracted = [r for r in result.results if r.status == "extracted"]
    if extracted:
        print(f"\n{'─' * 50}")
        print("EXTRACTED PROTOCOLS:")
        print(f"{'─' * 50}")
        for r in extracted:
            g = r.graph or {}
            n_nodes = len(g.get("nodes", []))
            n_edges = len(g.get("edges", []))
            ptype = g.get("protocol_type", "?")
            conf = g.get("extraction_confidence", "?")
            validated = g.get("_validated", False)
            tag = " [CORRECTED]" if validated else ""

            # Count node types
            types = {}
            for n in g.get("nodes", []):
                t = n.get("type", "?")
                types[t] = types.get(t, 0) + 1
            type_str = " ".join(f"{k}={v}" for k, v in sorted(types.items()))

            val_info = ""
            if r.validation:
                v = r.validation
                val_info = f" val={v.total_issues}issues"

            print(f"  {r.section.title}")
            print(f"    p.{r.section.start_page}-{r.section.end_page} | "
                  f"{ptype} | {n_nodes}n {n_edges}e | {type_str} | "
                  f"{r.tokens_used}tok {r.time_seconds:.1f}s{val_info}{tag}")

    # Errors
    errors = [r for r in result.results if r.status == "error"]
    if errors:
        print(f"\n{'─' * 50}")
        print("ERRORS:")
        print(f"{'─' * 50}")
        for r in errors:
            print(f"  {r.section.title}: {r.error}")

    # Validation summary
    validated = [r for r in result.results if r.validation is not None]
    if validated:
        print(f"\n{'─' * 50}")
        print("VALIDATION SUMMARY:")
        print(f"{'─' * 50}")
        corrected = sum(1 for r in validated if r.validation.was_corrected)
        clean = sum(1 for r in validated if r.validation.total_issues == 0)
        with_issues = len(validated) - clean
        val_tokens = sum(r.validation.validation_tokens for r in validated)
        print(f"  Validated:  {len(validated)}")
        print(f"  Clean:      {clean}")
        print(f"  With issues: {with_issues}")
        print(f"  Corrected:  {corrected}")
        print(f"  Val tokens: {val_tokens:,}")

    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "vic"

    config = BatchConfig(
        use_validator=True,
        output_dir="evaluation/extracted",
    )

    if target in ("vic", "both"):
        vic_result = run_and_summarize("VIC-BushfireHandbook.pdf", config)

    if target in ("irpg", "both"):
        # Use a separate output dir to avoid overwriting VIC results
        irpg_config = BatchConfig(
            use_validator=True,
            output_dir="evaluation/extracted",
        )
        irpg_result = run_and_summarize("IRPG-2025.pdf", irpg_config)
