"""
Parser Benchmark: Docling vs PyMuPDF

Compares text extraction quality for table-heavy protocol sections.
Evaluates: table structure preservation, section boundaries, processing time.

Usage:
    python evaluation/parser_benchmark.py <pdf_path> [--pages START END]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


def benchmark_pymupdf(pdf_path: str, start_page: int, end_page: int) -> dict:
    """Extract text using PyMuPDF and measure quality indicators."""
    from src.ingestion.pdf_ingest import ingest_pdf

    t0 = time.time()
    doc = ingest_pdf(pdf_path)
    elapsed = time.time() - t0

    parts = []
    table_count = 0
    for pg_num in range(start_page, end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        page = doc.pages[pg_num - 1]
        parts.append(page.full_text)
        table_count += len(page.tables)
        for table in page.tables:
            for row in table.cells:
                parts.append(" | ".join(row))

    text = "\n".join(parts)

    return {
        "parser": "PyMuPDF",
        "time_seconds": round(elapsed, 3),
        "text_length": len(text),
        "tables_detected": table_count,
        "has_pipe_delimiters": "|" in text,
        "text_preview": text[:500],
        "full_text": text,
    }


def benchmark_docling(pdf_path: str, start_page: int, end_page: int) -> dict:
    """Extract text using Docling and measure quality indicators."""
    try:
        from src.ingestion.docling_loader import extract_with_docling
    except ImportError as e:
        return {
            "parser": "Docling",
            "error": str(e),
            "time_seconds": 0,
            "text_length": 0,
        }

    t0 = time.time()
    text = extract_with_docling(pdf_path, start_page, end_page)
    elapsed = time.time() - t0

    # Count markdown tables (lines starting with |)
    table_lines = [l for l in text.split("\n") if l.strip().startswith("|")]

    return {
        "parser": "Docling",
        "time_seconds": round(elapsed, 3),
        "text_length": len(text),
        "markdown_table_lines": len(table_lines),
        "has_markdown_tables": len(table_lines) > 0,
        "text_preview": text[:500],
        "full_text": text,
    }


def compare(pdf_path: str, start_page: int, end_page: int) -> None:
    """Run both parsers and compare results."""
    print(f"Benchmarking: {Path(pdf_path).name} pages {start_page}-{end_page}")
    print("=" * 60)

    pymupdf_result = benchmark_pymupdf(pdf_path, start_page, end_page)
    print(f"\nPyMuPDF:")
    print(f"  Time: {pymupdf_result['time_seconds']}s")
    print(f"  Text length: {pymupdf_result['text_length']} chars")
    print(f"  Tables detected: {pymupdf_result['tables_detected']}")
    print(f"  Preview:\n    {pymupdf_result['text_preview'][:200]}...")

    print()

    docling_result = benchmark_docling(pdf_path, start_page, end_page)
    if "error" in docling_result:
        print(f"Docling: SKIPPED ({docling_result['error']})")
    else:
        print(f"Docling:")
        print(f"  Time: {docling_result['time_seconds']}s")
        print(f"  Text length: {docling_result['text_length']} chars")
        print(f"  Markdown table lines: {docling_result.get('markdown_table_lines', 0)}")
        print(f"  Preview:\n    {docling_result['text_preview'][:200]}...")

    print("\n" + "=" * 60)
    print("COMPARISON:")
    if "error" not in docling_result:
        speed_ratio = pymupdf_result["time_seconds"] / max(docling_result["time_seconds"], 0.001)
        print(f"  Speed: PyMuPDF is {speed_ratio:.1f}x {'faster' if speed_ratio > 1 else 'slower'}")
        print(f"  Text: PyMuPDF={pymupdf_result['text_length']} chars, "
              f"Docling={docling_result['text_length']} chars")
        print(f"  Tables: PyMuPDF detected {pymupdf_result['tables_detected']} tables, "
              f"Docling has {docling_result.get('markdown_table_lines', 0)} table lines")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation/parser_benchmark.py <pdf_path> [--pages START END]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    start_page = 1
    end_page = 5  # Default to first 5 pages

    if "--pages" in sys.argv:
        idx = sys.argv.index("--pages")
        start_page = int(sys.argv[idx + 1])
        end_page = int(sys.argv[idx + 2])

    compare(pdf_path, start_page, end_page)
