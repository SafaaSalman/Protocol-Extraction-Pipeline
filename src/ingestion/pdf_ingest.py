"""
PDF Ingestion Module

Extracts structured content from PDF documents using PyMuPDF:
- Per-page text blocks with bounding boxes
- Tables
- Page images (rendered)
- Document and page metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    block_type: str  # "text" or "image"
    page: int


@dataclass
class TableData:
    cells: list[list[str]]
    bbox: tuple[float, float, float, float]
    page: int


@dataclass
class PageContent:
    page_number: int  # 1-based
    width: float
    height: float
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    full_text: str = ""
    has_images: bool = False
    image_count: int = 0


@dataclass
class DocumentContent:
    filename: str
    page_count: int
    is_native: bool  # True if text-based PDF, False if scanned
    pages: list[PageContent] = field(default_factory=list)
    toc: list[dict] = field(default_factory=list)  # Table of contents entries


def ingest_pdf(pdf_path: str | Path) -> DocumentContent:
    """Extract structured content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentContent with per-page text blocks, tables, and metadata.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    # Extract table of contents
    raw_toc = doc.get_toc(simple=True)
    toc = [
        {"level": entry[0], "title": entry[1], "page": entry[2]}
        for entry in raw_toc
    ]

    pages: list[PageContent] = []
    total_text_length = 0
    total_pages_with_text = 0

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_content = _extract_page(page, page_idx + 1)
        pages.append(page_content)

        if page_content.full_text.strip():
            total_pages_with_text += 1
            total_text_length += len(page_content.full_text.strip())

    # Heuristic: if most pages have extractable text, it's a native PDF
    is_native = (total_pages_with_text / max(doc.page_count, 1)) > 0.5

    doc.close()

    return DocumentContent(
        filename=pdf_path.name,
        page_count=len(pages),
        is_native=is_native,
        pages=pages,
        toc=toc,
    )


def _extract_page(page: fitz.Page, page_number: int) -> PageContent:
    """Extract content from a single PDF page."""
    rect = page.rect
    content = PageContent(
        page_number=page_number,
        width=rect.width,
        height=rect.height,
    )

    # Extract text blocks with positions
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    text_parts = []

    for block in blocks:
        if block["type"] == 0:  # text block
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
                block_text += "\n"
            block_text = block_text.strip()
            if block_text:
                tb = TextBlock(
                    text=block_text,
                    bbox=(block["bbox"][0], block["bbox"][1],
                          block["bbox"][2], block["bbox"][3]),
                    block_type="text",
                    page=page_number,
                )
                content.text_blocks.append(tb)
                text_parts.append(block_text)
        elif block["type"] == 1:  # image block
            content.has_images = True
            content.image_count += 1

    content.full_text = "\n\n".join(text_parts)

    # Extract tables (PyMuPDF built-in table detection)
    try:
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            rows = table.extract()
            # Clean None values
            clean_rows = [
                [cell if cell is not None else "" for cell in row]
                for row in rows
            ]
            td = TableData(
                cells=clean_rows,
                bbox=tuple(table.bbox),
                page=page_number,
            )
            content.tables.append(td)
    except Exception:
        pass  # Table detection may fail on some pages

    return content


def render_page_image(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = 200,
) -> bytes:
    """Render a single page as a PNG image.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number.
        dpi: Resolution for rendering.

    Returns:
        PNG image bytes.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_number - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def save_document_json(doc_content: DocumentContent, output_path: str | Path) -> None:
    """Save extracted document content to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(doc_content)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(doc_content: DocumentContent) -> None:
    """Print a summary of the extracted document."""
    print(f"Document: {doc_content.filename}")
    print(f"Pages: {doc_content.page_count}")
    print(f"Native PDF: {doc_content.is_native}")
    print(f"TOC entries: {len(doc_content.toc)}")

    pages_with_tables = sum(1 for p in doc_content.pages if p.tables)
    pages_with_images = sum(1 for p in doc_content.pages if p.has_images)
    total_blocks = sum(len(p.text_blocks) for p in doc_content.pages)
    total_tables = sum(len(p.tables) for p in doc_content.pages)

    print(f"Total text blocks: {total_blocks}")
    print(f"Total tables: {total_tables}")
    print(f"Pages with tables: {pages_with_tables}")
    print(f"Pages with images: {pages_with_images}")

    if doc_content.toc:
        print("\nTable of Contents (first 20 entries):")
        for entry in doc_content.toc[:20]:
            indent = "  " * (entry["level"] - 1)
            print(f"  {indent}[p.{entry['page']}] {entry['title']}")
        if len(doc_content.toc) > 20:
            print(f"  ... and {len(doc_content.toc) - 20} more entries")
