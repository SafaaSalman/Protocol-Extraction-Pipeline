# Plan: Pipeline Improvement via Literature & Technology Integration

Improve Edge F1 from **0.432** (held-out) to **>0.70** by fixing edge extraction via better prompting, leveraging the **PET dataset** for few-shot examples, upgrading PDF parsing with **Docling**, and migrating graph storage to **Neo4j** for multi-use analysis. Organized into 6 phases — first 3 are critical path, last 3 are value-adds.

---

## Phase 1: Fix Edge Extraction (Critical — Highest Impact)

**Problem**: Held-out Edge F1 = 0.432. Decision branching under-detected, condition labels inaccurate, over-extraction creates synthetic intermediate nodes.

1. **Restructure SYSTEM_PROMPT edge-specific rules** — Replace vague rules 2 and 7 with explicit patterns, e.g.: *"For every if/then/else, create exactly one decision node with one edge per branch. Label each edge with the exact condition text."* Add anti-over-extraction rule: *"Do NOT decompose a single action into sub-steps unless the source explicitly lists them as separate items."* — Modify `SYSTEM_PROMPT` in `src/extraction/protocol_extractor.py`

2. **Add Triage as 3rd few-shot example** — *depends on 1* — Move `evaluation/gold/triage_01.json` (15n/20e, 5 decision nodes) from held-out to few-shot. It's the hardest branching graph and directly targets the weakness. Compensate by creating new golds (Phase 3).

3. **Add edge-focused self-review pass** — *parallel with 2* — After initial extraction, a second GPT-4o-mini call specifically reviews edge completeness: *"For each decision node, verify ≥2 outgoing edges with distinct conditions."* New function `review_edges()` in `src/extraction/protocol_extractor.py`

4. **Implement TextFlow intermediate representation for diagrams** — *parallel with 1-3* — Based on **Ye et al. 2024**: for `decision_tree`/`hybrid` sections, convert image → Mermaid text first → parse to protocol JSON. Current direct image→JSON misses arrow directions. Modify `extract_protocol_multimodal()` in `src/extraction/protocol_extractor.py`

**Expected Impact**: Edge F1 +0.15–0.25 (to ~0.60–0.68)

---

## Phase 2: Upgrade PDF Parsing (Medium Impact)

**Problem**: ~15 table-format sections poorly extracted; PyMuPDF doesn't preserve table structure.

5. **Evaluate Docling vs PyMuPDF head-to-head** — IBM Docling (MIT, DocLayNet + TableFormer) outputs Markdown with table structures. Benchmark on 5 PDFs comparing table preservation, section boundary detection, processing time. New script `evaluation/parser_benchmark.py`

6. **Integrate Docling for table sections** — *depends on 5* — If Docling wins on tables: new `src/ingestion/docling_loader.py`, route table-format sections through Docling, keep PyMuPDF for text sections (faster).

7. **Add table-specific extraction prompt** — *parallel with 6* — For `table` format: *"Each row represents a condition-action pair. Convert each row to: decision node (condition column) → action node (action column)."* New `TABLE_EXTRACTION_PROMPT` in `src/extraction/protocol_extractor.py`

**Expected Impact**: Table protocols from ~0 to F1 0.60+

---

## Phase 3: Evaluation & Gold Standard Expansion (Critical — Enables Measurement)

**Problem**: Only 3 held-out golds (after moving Triage); evaluation ignores edge conditions.

8. **Add edge condition comparison to evaluation** — Current `evaluate_edges()` in `evaluate.py` only compares `(from, to)` pairs, ignoring conditions. Add `SequenceMatcher` on condition labels. Report: Edge Connectivity F1 (current) + Edge Condition Accuracy (new).

9. **Create 3–5 new gold standards** — *parallel with 8* — Select from existing extractions covering gaps: 1 table-format (`aviation_user_checklist`), 1 long sequential (`hazmat_incident_operations`), 1 complex decision tree (`multi_casualty_triage_system`), 1 hybrid (`helicopter_extraction_operations`). Same schema as `evaluation/gold/`.

10. **Add graph-level evaluation metrics** — *depends on 8* — Path accuracy (% of gold start→end traversals present), branching accuracy (% of decision nodes with correct outgoing count), structural similarity (normalized graph edit distance). New functions in `evaluate.py`.

**Expected Impact**: Reliable eval with 6–8 held-out golds; condition accuracy reveals hidden quality issues.

---

## Phase 4: Neo4j Integration for Multi-Use Analysis (Value-Add)

**Problem**: Current graphs are isolated JSON files. No cross-protocol queries.

11. **Set up Neo4j + neo4j-graphrag-python** — `pip install neo4j neo4j-graphrag[openai]`, Docker or Neo4j Desktop (Community Edition, free). New `src/graph/neo4j_client.py`.

12. **Design Neo4j schema using GraphRAG SchemaBuilder** — *depends on 11* — Map JSON schema: `ProtocolNode` (id, type, text, page), relationships `NEXT_STEP`/`BRANCH_YES`/`BRANCH_NO`/`BRANCH_CONDITION`, meta-nodes `Protocol`/`Document`, cross-refs as `REFERENCES` relationships. New `src/graph/schema.py`.

13. **Load all 79 extracted protocols into Neo4j** — *depends on 12* — Use `Neo4jWriter` or Cypher batch insert. Add vector embeddings on node text for similarity search. New `src/graph/loader.py`.

14. **Build cross-protocol query layer** — *depends on 13* — Cypher queries: "Find all protocols referencing CPR", "Show decision paths for burns + not breathing", "Which protocols share common actions?" Use `VectorCypherRetriever` for natural language queries. New `src/graph/queries.py`.

**Expected Impact**: Enables "analyze graphs for multiple use"; cross-protocol traversal for thesis demo.

---

## Phase 5: PET Dataset Integration (Value-Add)

15. **Download and convert PET to our schema** — PET (patriziobellan/PET on HuggingFace, MIT license): 45 process descriptions annotated with activities, gateways, actors, flow relations. Convert: PET Activity → `action`, XOR Gateway → `decision`, flow → edge, Condition Specification → edge condition. New `src/evaluation/pet_converter.py`.

16. **Use PET examples as additional few-shot candidates** — *depends on 15* — Select 3–5 PET examples matching our complexity levels. A/B test: current 2 few-shots vs PET-augmented 4–5 examples.

17. **Benchmark against PET published baselines** — *depends on 15* — Neuberger 2024 tested 8 LLMs; compare our pipeline. New `evaluation/pet_benchmark.py`.

**Expected Impact**: External validation; better few-shot diversity; academic credibility.

---

## Phase 6: PyReason Temporal Reasoning (Optional/Future)

18. **Evaluate PyReason feasibility** — Pilot with CPR: encode as PyReason facts + rules. Can it answer "Patient not breathing, no pulse — what's the correct action sequence?"

19. **Define protocol execution rules** — If viable: encode traversal rules for protocol compliance checking.

**Expected Impact**: Differentiating thesis contribution if time permits.

---

## Relevant Files

| File | Changes |
|------|---------|
| `src/extraction/protocol_extractor.py` | SYSTEM_PROMPT rewrite (1), 3rd few-shot (2), `review_edges()` (3), TextFlow (4), table prompt (7) |
| `evaluate.py` | Condition comparison (8), path metrics (10) |
| `src/extraction/batch_extractor.py` | Docling router (6) |
| `evaluation/gold/` | 3–5 new gold standards (9) |
| `requirements.txt` | Add docling, neo4j, neo4j-graphrag, pyreason |
| **New**: `src/ingestion/docling_loader.py` | Docling PDF loader (6) |
| **New**: `src/graph/neo4j_client.py`, `schema.py`, `loader.py`, `queries.py` | Neo4j integration (11–14) |
| **New**: `src/evaluation/pet_converter.py`, `evaluation/pet_benchmark.py` | PET integration (15–17) |

---

## Verification

1. **After Phase 1**: `python evaluate.py` on held-out golds → Edge F1 ≥ 0.60
2. **After Phase 2**: `python evaluation/parser_benchmark.py` → table structure preserved ≥80%
3. **After Phase 3**: Full eval with new golds + conditions → ≥6 held-out golds, condition accuracy reported
4. **After Phase 4**: Cypher queries in Neo4j Browser → ≥3 cross-protocol queries working
5. **After Phase 5**: `python evaluation/pet_benchmark.py` → competitive with Neuberger 2024 (~0.80 activity F1, ~0.60 gateway F1)

---

## Decisions

- **Triage → few-shot**: It's the hardest branching example; compensated by 3–5 new golds
- **Docling supplements, not replaces, PyMuPDF**: PyMuPDF faster for text; Docling only for tables
- **Neo4j Community (free)**: Sufficient for thesis; no Enterprise features needed
- **PET domain gap acknowledged**: Business processes ≠ emergency protocols, but structural patterns (activities, gateways, flows) transfer

---

## Datasets Identified

1. **PET** — 45 annotated process descriptions, MIT, HuggingFace, token+relation annotations, baselines published
2. **GTWiki** — 176K Wikipedia text + Wikidata triples, MIT, general-domain (less relevant but usable for general text→graph validation)
3. **Gateway Extraction** (Janek7/gateway-extraction) — Built on PET, specifically targets XOR/AND gateway detection + sequence flow derivation

---

## Technology Assessment

### Neo4j GraphRAG Python (`neo4j-graphrag[openai]`)
- **Fit**: HIGH — directly addresses "analyze graphs for multiple use"
- **SimpleKGPipeline**: PDF → schema-guided extraction → Neo4j (but we already have extraction; just need storage + retrieval)
- **Schema-guided extraction**: Enforce node/edge types via SchemaBuilder
- **Entity resolution**: Fuzzy/semantic matching for cross-protocol entity deduplication
- **Retrievers**: VectorCypherRetriever enables "find protocols for X" queries
- **Custom KGWriter**: Can write JSON files instead of Neo4j for offline use
- **Risk**: Experimental API ("API changes and bug fixes are expected")
- **Recommendation**: Use Neo4j for storage + retrieval only. Keep our custom extraction pipeline.

### PyReason (v3.5.0)
- **Fit**: MEDIUM — interesting for protocol compliance reasoning but not core to extraction
- **Capability**: Temporal logic rules over graphs, propositional + first-order logic
- **Use case**: "Given these symptoms, what protocol applies?" → rule-based traversal
- **Risk**: Small community (333 stars), Python 3.7-3.10 only, numba dependency
- **Recommendation**: Phase 6 only. Pilot with one protocol first.

### Docling (IBM)
- **Fit**: HIGH for table protocols — directly solves table extraction weakness
- **Capability**: DocLayNet layout model + TableFormer; outputs Markdown with table structures
- **Risk**: Low — MIT license, IBM-backed, actively maintained
- **Recommendation**: Evaluate head-to-head with PyMuPDF on table sections.

---

## Further Considerations

1. **Few-shot prompt length**: Adding Triage as 3rd few-shot + PET examples may push prompt length beyond context window. Implement dynamic example selection — pick 2–3 most relevant examples based on detected format type (sequential → sequential examples, decision_tree → branching examples).

2. **Neo4j timing**: If thesis timeline is tight, Phase 4 can be simplified to just loading JSONs into Neo4j + basic Cypher queries, skipping GraphRAG retrieval. Basic queries alone demonstrate multi-use capability.

3. **PET domain gap**: Few-shot selection from PET should prioritize examples with imperative language and decision logic, not business-specific language like "sales department receives order."
