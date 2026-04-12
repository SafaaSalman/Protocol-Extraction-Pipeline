"""
Protocol Section Detector

Detects individual protocol sections within a PDF document using:
- Table of contents (primary signal)
- Heading/keyword heuristics (fallback for PDFs without TOC)
- Format classification per section
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.pdf_ingest import DocumentContent, PageContent


# Keywords that signal protocol/procedure content
PROTOCOL_KEYWORDS = re.compile(
    r"\b(protocol|procedure|algorithm|process|checklist|"
    r"guidelines|operations|assessment|response|safety|"
    r"management|system|reporting)\b",
    re.IGNORECASE,
)

# Patterns that suggest decision/branching logic
DECISION_PATTERNS = re.compile(
    r"\b(if\b.{0,30}\bthen\b|yes\s*/\s*no|"
    r"if\s+(the|patient|victim|fire|smoke|no|yes)|"
    r"determine\s+whether|"
    r"is\s+(the|patient|fire|smoke)\b.{0,30}\?|"
    r"not breathing|absent|present)\b",
    re.IGNORECASE,
)

# Patterns that suggest sequential numbered steps
NUMBERED_STEP_PATTERN = re.compile(
    r"^\s*(\d+)\.\s+\w",
    re.MULTILINE,
)

# Imperative verbs common in protocol actions
IMPERATIVE_PATTERN = re.compile(
    r"^\s*[•\-□]\s*(assess|check|monitor|remove|apply|call|"
    r"notify|evacuate|isolate|stabilize|transport|administer|"
    r"secure|identify|establish|contact|perform|maintain|"
    r"document|report|cool|elevate|splint|start|stop|open|"
    r"place|activate|obtain|provide|examine|look|ensure)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Cross-reference patterns
CROSS_REF_PATTERN = re.compile(
    r"(?:see|refer to|page|appendix)\s+[\w\d]+",
    re.IGNORECASE,
)


@dataclass
class ProtocolSection:
    title: str
    start_page: int  # 1-based inclusive
    end_page: int  # 1-based inclusive
    toc_level: int  # 1 = category, 2 = protocol
    category: str  # Parent category name
    format_hint: str  # "sequential" | "decision_tree" | "table" | "hybrid" | "checklist" | "reference"
    has_tables: bool = False
    has_numbered_steps: bool = False
    has_decision_logic: bool = False
    has_cross_references: bool = False
    is_protocol: bool = True  # False for non-protocol sections (notes, glossary, etc.)
    page_count: int = 0
    confidence: float = 0.0  # How confident we are this is a real protocol


@dataclass
class DetectionResult:
    sections: list[ProtocolSection]
    categories: list[str]
    total_protocols: int
    detection_method: str  # "toc" | "heuristic" | "hybrid"


def detect_sections(doc: DocumentContent) -> DetectionResult:
    """Detect protocol sections in a document.

    Uses TOC if available, falls back to heuristic detection.
    """
    if doc.toc:
        sections = _detect_from_toc(doc)
        method = "toc"
    else:
        sections = _detect_heuristic(doc)
        method = "heuristic"

    # Classify format and compute confidence for each section
    for section in sections:
        _classify_format(section, doc)
        _compute_confidence(section, doc)

    categories = sorted(set(s.category for s in sections))
    total_protocols = sum(1 for s in sections if s.is_protocol)

    return DetectionResult(
        sections=sections,
        categories=categories,
        total_protocols=total_protocols,
        detection_method=method,
    )


def _detect_from_toc(doc: DocumentContent) -> list[ProtocolSection]:
    """Detect sections using table of contents entries."""
    toc = doc.toc
    sections: list[ProtocolSection] = []

    # Identify which level-1 entries have children (level 2+)
    level1_has_children: set[int] = set()
    for i, entry in enumerate(toc):
        if entry["level"] == 1:
            # Check if any subsequent entry before the next level-1 is a child
            for j in range(i + 1, len(toc)):
                if toc[j]["level"] == 1:
                    break
                if toc[j]["level"] > 1:
                    level1_has_children.add(i)
                    break

    current_category = "Uncategorized"

    for i, entry in enumerate(toc):
        level = entry["level"]
        title = entry["title"].strip()
        start_page = entry["page"]

        # Update current category for level-1 entries
        if level == 1:
            current_category = _clean_category_name(title)

        # Determine end page: page before the next entry at same or higher level
        end_page = _find_section_end(toc, i, doc.page_count)

        # Ensure end_page >= start_page (can happen when two entries share a page)
        end_page = max(end_page, start_page)

        section = ProtocolSection(
            title=title,
            start_page=start_page,
            end_page=end_page,
            toc_level=level,
            category=current_category,
            format_hint="unknown",
            page_count=end_page - start_page + 1,
        )

        # Level-1 entries that have children are category headers, not protocols.
        # Level-1 entries WITHOUT children are standalone leaf protocols.
        if level == 1 and i in level1_has_children:
            section.is_protocol = False

        # Filter out known non-protocol sections
        if _is_non_protocol(title):
            section.is_protocol = False

        sections.append(section)

    return sections


def _find_section_end(
    toc: list[dict], current_idx: int, total_pages: int
) -> int:
    """Find the last page of a section based on the next TOC entry."""
    current_level = toc[current_idx]["level"]

    for j in range(current_idx + 1, len(toc)):
        next_level = toc[j]["level"]
        # Next entry at same or higher (lower number) level ends this section
        if next_level <= current_level:
            return toc[j]["page"] - 1

    # Last section extends to the end of the document
    return total_pages


def _detect_heuristic(doc: DocumentContent) -> list[ProtocolSection]:
    """Fallback: detect sections using text heuristics when no TOC is available."""
    sections: list[ProtocolSection] = []
    current_section: Optional[ProtocolSection] = None

    for page in doc.pages:
        heading = _detect_heading(page)

        if heading and PROTOCOL_KEYWORDS.search(heading):
            # Close previous section
            if current_section is not None:
                current_section.end_page = page.page_number - 1
                current_section.page_count = (
                    current_section.end_page - current_section.start_page + 1
                )
                sections.append(current_section)

            current_section = ProtocolSection(
                title=heading,
                start_page=page.page_number,
                end_page=page.page_number,
                toc_level=2,
                category="Detected",
                format_hint="unknown",
            )

    # Close final section
    if current_section is not None:
        current_section.end_page = doc.page_count
        current_section.page_count = (
            current_section.end_page - current_section.start_page + 1
        )
        sections.append(current_section)

    return sections


def _detect_heading(page: PageContent) -> Optional[str]:
    """Try to detect a heading/title from the top text blocks of a page."""
    # Look at the first few text blocks for short, title-like text
    for block in page.text_blocks[:5]:
        text = block.text.strip()
        # Skip very short text (page numbers, headers) and very long text (paragraphs)
        if 5 < len(text) < 120:
            # Skip common header/footer patterns
            if re.match(r"^(IRPG|Page|\d+\s+of\s+\d+)", text, re.IGNORECASE):
                continue
            # Title-like: starts with uppercase or has protocol keywords
            if text[0].isupper() and PROTOCOL_KEYWORDS.search(text):
                return text
    return None


def _clean_category_name(title: str) -> str:
    """Extract clean category name from a level-1 TOC entry."""
    # Remove parenthetical color references like "(green pages)"
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", title).strip()
    return cleaned


NON_PROTOCOL_TITLES = re.compile(
    r"\b(notes?|glossary|index|table of contents|"
    r"appendix|references|abbreviations|"
    r"acknowledgments?|foreword|preface|introduction)\b",
    re.IGNORECASE,
)


def _is_non_protocol(title: str) -> bool:
    """Check if a section title indicates non-protocol content."""
    return bool(NON_PROTOCOL_TITLES.search(title))


def _classify_format(section: ProtocolSection, doc: DocumentContent) -> None:
    """Classify the dominant format of a protocol section."""
    if not section.is_protocol:
        section.format_hint = "non_protocol"
        return

    # Gather text and stats from all pages in this section
    combined_text = ""
    table_count = 0
    total_blocks = 0

    for pg_num in range(section.start_page, section.end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        page = doc.pages[pg_num - 1]
        combined_text += page.full_text + "\n"
        table_count += len(page.tables)
        total_blocks += len(page.text_blocks)

    # Check for features
    numbered_steps = NUMBERED_STEP_PATTERN.findall(combined_text)
    decision_matches = DECISION_PATTERNS.findall(combined_text)
    imperative_matches = IMPERATIVE_PATTERN.findall(combined_text)
    cross_refs = CROSS_REF_PATTERN.findall(combined_text)

    section.has_tables = table_count > 0
    section.has_numbered_steps = len(numbered_steps) >= 2
    section.has_decision_logic = len(decision_matches) >= 1
    section.has_cross_references = len(cross_refs) >= 1

    # Classify based on dominant feature
    has_steps = section.has_numbered_steps
    has_decisions = section.has_decision_logic
    has_tables = section.has_tables
    has_bullets = len(imperative_matches) >= 3

    if has_tables and has_decisions:
        section.format_hint = "hybrid"
    elif has_tables and not has_steps and not has_decisions:
        section.format_hint = "table"
    elif has_steps and has_decisions:
        section.format_hint = "decision_tree"
    elif has_steps and not has_decisions:
        section.format_hint = "sequential"
    elif has_bullets and not has_steps:
        section.format_hint = "checklist"
    elif has_decisions:
        section.format_hint = "decision_tree"
    else:
        section.format_hint = "reference"


def _compute_confidence(
    section: ProtocolSection, doc: DocumentContent
) -> None:
    """Compute confidence that this section is a real, extractable protocol."""
    if not section.is_protocol:
        section.confidence = 0.0
        return

    score = 0.0

    # Title contains protocol keywords
    if PROTOCOL_KEYWORDS.search(section.title):
        score += 0.25

    # Has structured content (steps, decisions, tables)
    if section.has_numbered_steps:
        score += 0.2
    if section.has_decision_logic:
        score += 0.2
    if section.has_tables:
        score += 0.1

    # Reasonable length (1-10 pages is typical for a single protocol)
    if 1 <= section.page_count <= 10:
        score += 0.15
    elif section.page_count > 10:
        score += 0.05  # Might be a large section, not a single protocol

    # Has imperative action language
    combined_text = ""
    for pg_num in range(section.start_page, section.end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        combined_text += doc.pages[pg_num - 1].full_text + "\n"

    if IMPERATIVE_PATTERN.findall(combined_text):
        score += 0.1

    section.confidence = min(score, 1.0)


def print_detection_summary(result: DetectionResult) -> None:
    """Print a readable summary of detected sections."""
    print(f"Detection method: {result.detection_method}")
    print(f"Categories: {len(result.categories)}")
    print(f"Total sections: {len(result.sections)}")
    print(f"Protocol sections: {result.total_protocols}")
    print()

    for cat in result.categories:
        cat_sections = [s for s in result.sections if s.category == cat]
        protocols = [s for s in cat_sections if s.is_protocol]
        print(f"--- {cat} ({len(protocols)} protocols) ---")
        for s in cat_sections:
            marker = "  " if s.is_protocol else "  [skip] "
            conf = f"conf={s.confidence:.2f}" if s.is_protocol else ""
            print(
                f"{marker}{s.title}"
                f"  (p.{s.start_page}-{s.end_page}, "
                f"{s.format_hint}, {conf})"
            )
        print()
