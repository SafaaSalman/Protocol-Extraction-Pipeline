"""
Batch Protocol Extraction with Confidence Gating

Extracts protocols from a full PDF using:
- Confidence-gated filtering: skip low-value reference sections
- TOC context enrichment: provide category breadcrumbs + sibling context
- Lightweight pre-classification for borderline sections
- Optional multimodal extraction for diagram-heavy pages
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.ingestion.pdf_ingest import DocumentContent, ingest_pdf
from src.extraction.section_detector import (
    DetectionResult,
    ProtocolSection,
    detect_sections,
)
from src.extraction.protocol_extractor import (
    extract_protocol,
    extract_protocol_multimodal,
    save_extraction,
)
from src.extraction.validator import validate_protocol, ValidationResult


# ── Confidence thresholds ────────────────────────────────────────────

DEFAULT_EXTRACT_THRESHOLD = 0.35   # Extract without question
DEFAULT_CLASSIFY_THRESHOLD = 0.15  # Below this → skip entirely
# Between classify and extract → run a cheap pre-classification call


@dataclass
class BatchConfig:
    """Configuration for a batch extraction run."""
    extract_threshold: float = DEFAULT_EXTRACT_THRESHOLD
    classify_threshold: float = DEFAULT_CLASSIFY_THRESHOLD
    use_multimodal: bool = False       # Use page images for all sections
    multimodal_formats: tuple = ("hybrid", "decision_tree")  # Formats that get multimodal
    use_few_shot: bool = True
    use_validator: bool = False        # Run semantic validation after extraction
    validator_model: str = "gpt-4o"    # Model for validation (can differ from extraction)
    model: str = "gpt-4o"
    dpi: int = 200
    output_dir: str = "evaluation/extracted"
    dry_run: bool = False              # If True, just report what would be extracted


@dataclass
class ExtractionResult:
    """Result for a single section extraction attempt."""
    section: ProtocolSection
    status: str        # "extracted" | "skipped_low_conf" | "skipped_classify" | "error"
    graph: Optional[dict] = None
    error: Optional[str] = None
    tokens_used: int = 0
    time_seconds: float = 0.0
    validation: Optional[ValidationResult] = None


@dataclass
class BatchResult:
    """Summary of a full batch extraction run."""
    pdf_name: str
    total_sections: int
    total_protocols: int
    extracted: int = 0
    skipped_low_conf: int = 0
    skipped_classify: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    results: list[ExtractionResult] = field(default_factory=list)


# ── Pre-classification ───────────────────────────────────────────────

PRE_CLASSIFY_PROMPT = """\
You are evaluating whether a section from an emergency management PDF contains \
an extractable protocol, procedure, or process with discrete steps.

Answer with a JSON object:
{
  "is_extractable": true/false,
  "reason": "<one sentence explanation>"
}

Criteria for extractable:
- Has numbered steps, bullet-point actions, or a clear sequential/branching process
- Describes HOW to do something (procedure), not just WHAT something is (description)
- Contains imperative language (assess, check, notify, evacuate, etc.)
- Has decision points, conditional logic, or a flowchart

NOT extractable:
- Pure definitions or descriptions with no actionable steps
- Lists of roles/responsibilities without a process flow
- Contact lists, glossaries, or reference tables with no procedure
- Background context or policy statements

Return ONLY valid JSON."""


def pre_classify_section(
    section: ProtocolSection,
    doc: DocumentContent,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> bool:
    """Cheap LLM call to decide if a borderline section is worth extracting.

    Uses gpt-4o-mini for cost efficiency (~10x cheaper than gpt-4o).
    """
    # Gather a text sample (first 1500 chars to keep tokens low)
    text_parts = []
    for pg_num in range(section.start_page, section.end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        text_parts.append(doc.pages[pg_num - 1].full_text)
    sample_text = "\n".join(text_parts)[:1500]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PRE_CLASSIFY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Section title: {section.title}\n"
                    f"Category: {section.category}\n"
                    f"Pages: {section.start_page}-{section.end_page}\n"
                    f"Detected features: tables={section.has_tables}, "
                    f"numbered_steps={section.has_numbered_steps}, "
                    f"decision_logic={section.has_decision_logic}\n\n"
                    f"--- TEXT SAMPLE ---\n{sample_text}"
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("is_extractable", False)


# ── TOC context enrichment ───────────────────────────────────────────

def build_toc_context(
    section: ProtocolSection,
    detection: DetectionResult,
) -> str:
    """Build a context string showing where this section sits in the document hierarchy.

    Provides:
    - Category breadcrumb
    - Sibling section titles (for cross-reference awareness)
    """
    # Find sibling sections (same category, is_protocol)
    siblings = [
        s for s in detection.sections
        if s.category == section.category
        and s.is_protocol
        and s.title != section.title
    ]

    lines = [f"Document context: {section.category} > {section.title}"]

    if siblings:
        sibling_titles = [s.title for s in siblings[:10]]  # Cap at 10
        lines.append(f"Related sections in this category: {', '.join(sibling_titles)}")

    return "\n".join(lines)


# ── Batch extraction ─────────────────────────────────────────────────

def plan_batch(
    detection: DetectionResult,
    config: BatchConfig,
) -> tuple[list[ProtocolSection], list[ProtocolSection], list[ProtocolSection]]:
    """Partition detected protocols into extract / classify / skip buckets.

    Returns:
        (to_extract, to_classify, to_skip)
    """
    to_extract = []
    to_classify = []
    to_skip = []

    for section in detection.sections:
        if not section.is_protocol:
            to_skip.append(section)
            continue

        if section.confidence >= config.extract_threshold:
            to_extract.append(section)
        elif section.confidence >= config.classify_threshold:
            to_classify.append(section)
        else:
            to_skip.append(section)

    return to_extract, to_classify, to_skip


def run_batch(
    pdf_path: str | Path,
    config: Optional[BatchConfig] = None,
) -> BatchResult:
    """Run confidence-gated batch extraction on a full PDF.

    Pipeline:
    1. Ingest PDF → detect sections
    2. Partition into extract / classify / skip
    3. Pre-classify borderline sections (cheap gpt-4o-mini call)
    4. Extract qualifying sections (gpt-4o, optionally multimodal)
    5. Save results and return summary
    """
    if config is None:
        config = BatchConfig()

    pdf_path = Path(pdf_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ingest & detect ──────────────────────────────────────
    print(f"[batch] Ingesting {pdf_path.name}...")
    doc = ingest_pdf(str(pdf_path))
    detection = detect_sections(doc)

    batch = BatchResult(
        pdf_name=pdf_path.name,
        total_sections=len(detection.sections),
        total_protocols=detection.total_protocols,
    )

    # ── Step 2: Partition ────────────────────────────────────────────
    to_extract, to_classify, to_skip = plan_batch(detection, config)

    print(f"[batch] {len(to_extract)} above threshold (>={config.extract_threshold})")
    print(f"[batch] {len(to_classify)} borderline (classify first)")
    print(f"[batch] {len(to_skip)} skipped (low confidence or non-protocol)")

    batch.skipped_low_conf = len(to_skip)

    if config.dry_run:
        _print_plan(to_extract, to_classify, to_skip)
        return batch

    client = OpenAI()

    # ── Step 3: Pre-classify borderline sections ─────────────────────
    promoted = 0
    for section in to_classify:
        try:
            is_extractable = pre_classify_section(section, doc, client)
            if is_extractable:
                to_extract.append(section)
                promoted += 1
            else:
                batch.skipped_classify += 1
                batch.results.append(ExtractionResult(
                    section=section,
                    status="skipped_classify",
                ))
        except Exception as e:
            # If pre-classification fails, include it for extraction anyway
            to_extract.append(section)
            promoted += 1

    if to_classify:
        print(f"[batch] Pre-classification: {promoted} promoted, "
              f"{batch.skipped_classify} filtered out")

    # ── Step 4: Extract ──────────────────────────────────────────────
    print(f"[batch] Extracting {len(to_extract)} protocols...")

    for i, section in enumerate(to_extract, 1):
        toc_context = build_toc_context(section, detection)
        print(f"  [{i}/{len(to_extract)}] {section.title} "
              f"(p.{section.start_page}-{section.end_page}, "
              f"conf={section.confidence:.2f}, {section.format_hint})")

        start_time = time.time()
        try:
            # Decide text-only vs multimodal
            use_mm = (
                config.use_multimodal
                or section.format_hint in config.multimodal_formats
            )

            if use_mm:
                graph = extract_protocol_multimodal(
                    section, doc, pdf_path,
                    model=config.model,
                    use_few_shot=config.use_few_shot,
                    dpi=config.dpi,
                    toc_context=toc_context,
                )
            else:
                graph = extract_protocol(
                    section, doc,
                    model=config.model,
                    use_few_shot=config.use_few_shot,
                    toc_context=toc_context,
                )

            # Inject TOC context metadata
            graph["_toc_context"] = toc_context

            elapsed = time.time() - start_time
            tokens = graph.get("_extraction_meta", {}).get("total_tokens", 0)

            # ── Optional: Validate extracted graph ───────────────────
            validation = None
            if config.use_validator:
                print(f"    -> validating...")
                val_start = time.time()
                section_text = _gather_section_text_for_validation(section, doc)
                validation = validate_protocol(
                    graph, section_text,
                    run_semantic=True,
                    model=config.validator_model,
                )
                val_elapsed = time.time() - val_start
                val_tokens = validation.validation_tokens
                tokens += val_tokens
                elapsed += val_elapsed

                if validation.was_corrected and validation.corrected_graph:
                    # Preserve extraction metadata on the corrected graph
                    validation.corrected_graph["_extraction_meta"] = graph.get("_extraction_meta")
                    validation.corrected_graph["_toc_context"] = toc_context
                    validation.corrected_graph["_validated"] = True
                    validation.corrected_graph["_correction_summary"] = (
                        validation.semantic_result.get("correction_summary")
                    )
                    graph = validation.corrected_graph
                    print(f"    -> corrected! ({val_tokens} val tokens, {val_elapsed:.1f}s)")
                else:
                    n_issues = validation.total_issues
                    print(f"    -> {'clean' if n_issues == 0 else str(n_issues) + ' issues'} "
                          f"({val_tokens} val tokens, {val_elapsed:.1f}s)")

            batch.extracted += 1
            batch.total_tokens += tokens
            batch.total_time += elapsed

            result = ExtractionResult(
                section=section,
                status="extracted",
                graph=graph,
                tokens_used=tokens,
                time_seconds=elapsed,
                validation=validation,
            )
            batch.results.append(result)

            # Save individual extraction
            safe_name = section.title.lower()
            safe_name = "".join(c if c.isalnum() or c == " " else "" for c in safe_name)
            safe_name = "_".join(safe_name.split())
            out_path = output_dir / f"{safe_name}.json"
            save_extraction(graph, out_path)
            print(f"    -> saved {out_path.name} ({tokens} tokens, {elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            batch.errors += 1
            batch.total_time += elapsed
            batch.results.append(ExtractionResult(
                section=section,
                status="error",
                error=str(e),
                time_seconds=elapsed,
            ))
            print(f"    -> ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n[batch] Done: {batch.extracted} extracted, "
          f"{batch.skipped_low_conf + batch.skipped_classify} skipped, "
          f"{batch.errors} errors")
    print(f"[batch] Total tokens: {batch.total_tokens:,}")
    print(f"[batch] Total time: {batch.total_time:.1f}s")

    return batch


def _gather_section_text_for_validation(
    section: ProtocolSection,
    doc: DocumentContent,
) -> str:
    """Gather section text for the validator (same logic as protocol_extractor)."""
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


def _print_plan(
    to_extract: list[ProtocolSection],
    to_classify: list[ProtocolSection],
    to_skip: list[ProtocolSection],
) -> None:
    """Print a dry-run plan showing what would happen."""
    print("\n── DRY RUN PLAN ──\n")

    if to_extract:
        print(f"EXTRACT ({len(to_extract)} sections):")
        for s in to_extract:
            print(f"  + p.{s.start_page}-{s.end_page} [{s.format_hint}] "
                  f"conf={s.confidence:.2f} {s.title}")

    if to_classify:
        print(f"\nCLASSIFY FIRST ({len(to_classify)} sections):")
        for s in to_classify:
            print(f"  ? p.{s.start_page}-{s.end_page} [{s.format_hint}] "
                  f"conf={s.confidence:.2f} {s.title}")

    if to_skip:
        print(f"\nSKIP ({len(to_skip)} sections):")
        for s in to_skip:
            reason = "non-protocol" if not s.is_protocol else "low confidence"
            print(f"  x p.{s.start_page}-{s.end_page} [{s.format_hint}] "
                  f"conf={s.confidence:.2f} {s.title} ({reason})")
