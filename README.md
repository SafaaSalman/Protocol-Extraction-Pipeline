# Protocol Extraction Pipeline

An LLM-based pipeline that extracts structured protocol graphs from
emergency response PDF documents. Each extracted protocol is a
directed graph in which nodes represent procedural steps or decision
points and edges represent ordering and conditional branching. Every
node links back to the source page and text it came from.

The pipeline is exercised on two complete documents:

| Document               | Pages | 
| ---------------------- | ----- | 
| IRPG 2025              | 140   | 
| VIC Bushfire Handbook  | 100   | 

A total of 79 protocol graphs are produced across the two documents.
Five IRPG protocols (CPR, Burn Injuries, Patient Assessment, Risk
Management, Multi-Casualty Triage) are manually annotated as gold
standards and serve as the quantitative evaluation set.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# Optional: enables table-aware parsing
pip install docling

# 2. Set the OpenAI key (or put it in a local .env file)
export OPENAI_API_KEY=sk-...

# 3. Run extraction on both PDFs
python run_full_batch.py both

# 4. Evaluate against the five gold standards
python eval_batch.py

# 5. Build the viewer index and open it in a browser
python generate_manifest.py
# then open viewer.html
```

## Pipeline stages

1. **PDF ingestion** — PyMuPDF for text and TOC; Docling for
   table-heavy sections.
2. **Section detection** — TOC-based, with a heuristic fallback
   for documents without a usable TOC.
3. **Confidence gating** — three-tier gate (`extract` / `pre-classify`
   / `skip`) that saves 50–70% of the LLM budget without losing
   extractable content.
4. **LLM extraction** — GPT-4o with a 21-rule system prompt, three
   few-shot examples, TOC context, and optional image input via a
   Mermaid intermediate representation.
5. **Edge review** — focused GPT-4o-mini second pass that fixes
   missing branches and condition labels.
6. **Validation** — eight deterministic structural checks plus a
   GPT-4o semantic review against the source text.
7. **Output** — one JSON file per protocol in
   `evaluation/extracted/`, plus a manifest for the viewer.

## Project layout

```
DataPipeline/
├── src/
│   ├── ingestion/       PDF loaders (PyMuPDF, Docling)
│   ├── extraction/      Section detection, batch orchestration,
│   │                    LLM extraction, validation
│   └── graph/           Neo4j export, KG enrichment, dataset adapters
├── evaluation/
│   ├── gold/            Five hand-annotated gold protocols
│   ├── extracted/       Pipeline output + manifest.json
│   ├── pet_benchmark.py
│   ├── brex_benchmark.py
│   └── parser_benchmark.py
├── schema/protocol_schema.json
├── run_full_batch.py    Primary entry point
├── eval_batch.py        Compare against gold standards
├── evaluate.py          Metrics library
├── load_neo4j.py        Load extracted graphs into Neo4j
├── visualize.py         Render graphs to PNG
├── viewer.html          Interactive web viewer
└── docker-compose.yml   Neo4j 5 Community service
```

## Optional: load into Neo4j

```bash
docker compose up -d
python load_neo4j.py --password protocol123 --stats
```

Add `--enrich` to run the optional KG enrichment pass that annotates
each protocol with roles, equipment, hazards, conditions, and
incident types.

## Optional: run external benchmarks

```bash
python evaluation/pet_benchmark.py    # Cross-domain (English BPM)
python evaluation/brex_benchmark.py   # Cross-lingual (Chinese rules)
```

## Output schema

Every extracted protocol conforms to `schema/protocol_schema.json`:

```json
{
  "protocol_id": "cpr",
  "title": "CPR",
  "source_pdf": "IRPG-2025.pdf",
  "pages": [127],
  "protocol_type": "decision_tree",
  "extraction_confidence": 0.85,
  "cross_references": ["Patient Assessment"],
  "nodes": [
    { "id": "n1", "type": "start", "text": "Ensure scene safety",
      "evidence": { "page": 127, "source_text": "..." } }
  ],
  "edges": [
    { "from": "n1", "to": "n2" },
    { "from": "n3", "to": "n4", "condition": "Yes (breathing)" }
  ]
}
```

Node types: `start`, `end`, `action`, `decision`. Edges optionally
carry a `condition` label for decision branches.

## Documentation

Full architecture rationale, design decisions, and per-module code
documentation are in `code_documentation.tex` (build with
`pdflatex --shell-escape`).

## Author

Safaa Salman — American University of Beirut
