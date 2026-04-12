"""
Docling-based PDF Loader

Alternative to PyMuPDF for PDFs with table-heavy protocols.
IBM Docling (MIT license) uses DocLayNet layout model + TableFormer
for superior table structure preservation.

Used selectively for sections classified as 'table' format,
while PyMuPDF remains the default for text-heavy sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def extract_with_docling(
    pdf_path: str | Path,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> str:
    """Extract text from a PDF using Docling, returning Markdown with table structures.

    Args:
        pdf_path: Path to the PDF file.
        start_page: 1-based start page (None = from beginning).
        end_page: 1-based end page (None = to end).

    Returns:
        Markdown-formatted text with tables preserved as Markdown tables.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "docling is required for table-aware PDF extraction. "
            "Install with: pip install docling"
        )

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    # Export as Markdown (preserves table structure)
    full_md = result.document.export_to_markdown()

    if start_page is None and end_page is None:
        return full_md

    # If page filtering is requested, extract page-specific content
    # Docling's document model provides page-level access
    parts = []
    for item in result.document.iterate_items():
        # Get the page number from the item's provenance
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            page_num = prov[0].page_no  # 1-based
            if start_page and page_num < start_page:
                continue
            if end_page and page_num > end_page:
                continue
        text = getattr(item, "export_to_markdown", lambda: str(item))()
        if text:
            parts.append(text)

    return "\n\n".join(parts) if parts else full_md


def extract_tables_from_pages(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
) -> list[dict]:
    """Extract structured table data from specific pages using Docling.

    Returns:
        List of dicts with keys: page, headers, rows
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.types.doc import TableItem as TableElement
    except ImportError:
        raise ImportError(
            "docling is required for table extraction. "
            "Install with: pip install docling"
        )

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    tables = []
    for item in result.document.iterate_items():
        if not isinstance(item, TableElement):
            continue

        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            page_num = prov[0].page_no
            if page_num < start_page or page_num > end_page:
                continue
        else:
            continue

        # Extract table as markdown then parse rows
        table_md = item.export_to_markdown()
        lines = [l.strip() for l in table_md.strip().split("\n") if l.strip()]

        headers = []
        rows = []
        for i, line in enumerate(lines):
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if i == 0:
                headers = cells
            elif line.startswith("|--") or line.startswith("| --"):
                continue  # Skip separator line
            else:
                rows.append(cells)

        tables.append({
            "page": page_num if prov else None,
            "headers": headers,
            "rows": rows,
            "markdown": table_md,
        })

    return tables
