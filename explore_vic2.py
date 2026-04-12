import sys
sys.path.insert(0, ".")
from src.ingestion.pdf_ingest import ingest_pdf
import fitz

pdf = fitz.open("VIC-BushfireHandbook.pdf")
toc = pdf.get_toc()
for e in toc[80:]:
    indent = "  " * (e[0] - 1)
    print(f"{indent}[L{e[0]}] p.{e[2]}: {e[1]}")

doc = ingest_pdf("VIC-BushfireHandbook.pdf")

for i in [28, 29, 38, 39, 43, 44, 55, 56, 72, 73, 74, 75, 76, 77]:
    pg = i - 1
    if pg < doc.page_count:
        print(f"\n{'='*60}")
        print(f"PAGE {i}")
        print(f"{'='*60}")
        print(doc.pages[pg].full_text[:700])

# Also look at table content on some key pages
for i in [39, 43, 44, 63, 64, 75]:
    pg = i - 1
    if pg < doc.page_count and doc.pages[pg].tables:
        print(f"\n--- TABLE on page {i} ---")
        for t in doc.pages[pg].tables:
            for row in t.cells[:8]:
                print(" | ".join(row))
            if len(t.cells) > 8:
                print(f"  ... {len(t.cells)} rows total")

pdf.close()
