"""Explore VIC Bushfire Handbook PDF structure."""
import sys
sys.path.insert(0, ".")
from src.ingestion.pdf_ingest import ingest_pdf
import fitz

doc = ingest_pdf("VIC-BushfireHandbook.pdf")
print(f"Pages: {doc.page_count}")

pdf = fitz.open("VIC-BushfireHandbook.pdf")
toc = pdf.get_toc()
print(f"TOC entries: {len(toc)}")
for e in toc[:80]:
    indent = "  " * (e[0] - 1)
    print(f"  {indent}[L{e[0]}] p.{e[2]}: {e[1]}")
if len(toc) > 80:
    print(f"  ... +{len(toc) - 80} more")

print(f"\n--- Tables ---")
table_pages = [(i+1, len(p.tables)) for i, p in enumerate(doc.pages) if p.tables]
print(f"Pages with tables: {len(table_pages)}")
for pg, c in table_pages[:20]:
    print(f"  p.{pg}: {c} table(s)")

print(f"\n--- Images ---")
img_pages = [(i+1, p.image_count) for i, p in enumerate(doc.pages) if p.image_count > 0]
print(f"Pages with images: {len(img_pages)}")

# Sample pages
for i in [0, 3, 5, 10, 15, 20, 30, 40, 50, 60]:
    if i < doc.page_count:
        print(f"\n{'='*60}")
        print(f"PAGE {i+1}")
        print(f"{'='*60}")
        print(doc.pages[i].full_text[:600])

pdf.close()
