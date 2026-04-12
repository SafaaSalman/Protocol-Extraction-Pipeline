Critique of the Conversation
What's good:

Core framing is correct — treating this as "protocol reconstruction" rather than text extraction is the right lens. The emphasis on preserving branching logic, not just extracting words, is the single most important insight.

The intermediate representation (decision graph with nodes/edges/conditions) is well-designed. The JSON schema proposed is clean and practical.

The two-layer architecture (Layer A: execution flow, Layer B: semantic knowledge) is academically strong. It gives you a protocol graph that preserves exact procedural order and a knowledge graph that enables cross-protocol reasoning.

Evidence grounding — linking every extracted node back to source page/bbox/text is critical for a thesis. This was correctly highlighted.

The "don't send 100 pages to one LLM call" warning is practical and correct. Chunking by protocol section first is essential.

Scoping advice (start with native PDFs, English, 2-3 formats) is sensible for thesis-sized work.

What's wrong or overengineered:

The 11-step pipeline is too ambitious for a thesis prototype. It reads like a product spec, not a research plan. You'd spend all your time building infrastructure and have nothing left for evaluation. The LLM describes 5 stages, then 11 steps, then 7 steps for "v1" — scope keeps shifting.

"Graph database / Neo4j / RDF/OWL" is premature. For a thesis prototype, you don't need a graph database. NetworkX + JSON files will do. The LLM mentioned this but buried it. This should be the default, not an alternative.

The page classifier (text/diagram/table/non-protocol) is presented as a simple step, but it's actually a research problem in itself. The conversation doesn't address how to build this classifier. What model? What training data? What features? This is hand-waved.

No concrete tool/library recommendations. The conversation never mentions PyMuPDF, pdfplumber, Docling, Marker, Unstructured, or any specific vision models. It stays entirely abstract.

The "three separate extractors" model (text/table/diagram) sounds clean but creates a hard integration problem. In real PDFs like the IRPG, a single protocol section often mixes text steps, tables, and inline diagrams on the same page. You'd need a way to merge partial extractions, which isn't addressed.

The knowledge graph enrichment (Phase 2) is under-specified. Extracting entities like "responder role" or "equipment" from protocol text is effectively a Named Entity Recognition problem on a domain-specific corpus. The conversation doesn't acknowledge this difficulty.

Validation is described generically. "Check every decision node has at least 2 outgoing paths" is trivial to implement; the hard validation is whether the content is correct, which requires human evaluation or gold-standard annotations.

What's missing:

No discussion of using modern document understanding models — tools like Docling (IBM), Marker, or multimodal LLMs (GPT-4o, Claude with vision) that can process page images directly. These dramatically simplify the "diagram extraction" problem.

No mention of prompt engineering strategy — if you're using an LLM to extract structured JSON from protocol sections, the prompt design is half the work. Zero-shot vs few-shot vs chain-of-thought matters a lot.

No evaluation baseline or gold standard. How will you measure success? The conversation mentions evaluation metrics but doesn't discuss how to create ground truth annotations.

No cost/latency analysis. Processing 100+ page PDFs through vision LLMs page-by-page gets expensive fast. This affects architecture decisions.

*No discussion of how protocols that span pages or reference other protocols ("see Appendix B") are handled concretely.

Refined Plan
Plan: Protocol Knowledge Graph Extraction from PDFs
Extract emergency/incident protocols from multi-format PDFs into structured decision graphs, then enrich into a protocol-centric knowledge graph. Focus on a thesis-quality prototype with measurable evaluation over the IRPG-2025 and 2-3 similar documents.

Phase 1: Ground Truth & Schema Definition (do first, blocks everything)

Manually annotate 5-8 protocols from IRPG-2025 as gold-standard JSON graphs — choose a mix of text-step protocols, table protocols, and at least one flowchart/decision tree
Define the protocol graph JSON schema formally (node types, edge types, metadata fields, evidence fields). Use the schema from the conversation as a starting point but add: protocol_type enum (sequential | decision_tree | table | hybrid), extraction_confidence, and cross_references
Define the knowledge graph ontology — node types (Protocol, Step, Decision, Action, Role, Equipment, Hazard, Incident) and relation types (HAS_STEP, NEXT, BRANCHES_TO, REQUIRES, PERFORMED_BY, APPLIES_TO). Keep it small — 8-10 node types, 10-12 relation types max

Phase 2: PDF Ingestion & Section Detection (parallel with Phase 1 schema work)

Use PyMuPDF (fitz) for text/layout extraction and pdf2image for page rendering — extract per-page text blocks, tables, and page images
Build a protocol section detector using heading/keyword heuristics (look for "Protocol", "Procedure", "Algorithm", numbered step patterns, imperative verbs) combined with ToC parsing if available. Output: list of {title, start_page, end_page, format_hint} per protocol
For each detected section, classify format as text_steps | table | diagram | mixed — start with rule-based heuristics (presence of arrows/shapes → diagram, grid structure → table, numbered lines → text steps)
Phase 3: Protocol Structure Extraction (depends on Phase 1 schema + Phase 2 sections)

Text/table protocols: Use an LLM (Claude or GPT-4) with a structured output prompt — feed the extracted text/table content for a section + the JSON schema + 2-3 few-shot examples from your gold standard → get back structured protocol graph JSON
Diagram protocols: Use a multimodal LLM (Claude with vision / GPT-4o) — feed the page image(s) + schema + few-shot examples → get back structured protocol graph JSON. This sidesteps the entire "detect boxes, detect arrows, OCR node text" sub-pipeline — modern vision models handle this directly
Cross-page continuity: For protocols spanning multiple pages, concatenate text or provide multiple images in sequence to the LLM with explicit instruction to treat them as one protocol
Phase 4: Knowledge Graph Enrichment (depends on Phase 3)

Run a second LLM pass over each extracted protocol to identify semantic entities (roles, equipment, hazards, conditions, incident types) and relations. Use the ontology from step 3 as the extraction schema
Link entities across protocols — e.g., "Paramedic" role appearing in multiple protocols gets merged into one KG node with edges to each protocol
Store everything in NetworkX (in-memory graph) + export to JSON-LD or simple JSON for persistence. No graph database needed
Phase 5: Validation & Evaluation (depends on Phase 3 + Phase 1 gold standard)

Structural validation (automated): every protocol has start node, decision nodes have ≥2 outgoing edges, no orphan nodes, all edges point to existing nodes
Content accuracy (against gold standard): compute node-level precision/recall (did the system find the right steps?), edge-level precision/recall (did it find the right connections?), and protocol boundary detection F1
Confidence scoring: assign confidence based on extraction method (text > table > diagram) and LLM self-reported uncertainty
Human evaluation: have 1-2 domain-knowledgeable reviewers score 10-15 extracted protocols on correctness (1-5 scale) and completeness (1-5 scale)
Phase 6: Presentation Layer (optional, depends on Phase 4)

Simple visualization: render extracted protocol graphs using Graphviz or pyvis alongside the source PDF page image for side-by-side comparison
Export knowledge graph to a queryable format — support basic queries like "show all protocols involving evacuation" or "what equipment does protocol X require"
Relevant files

IRPG-2025.pdf — primary test document, native digital PDF, wildfire incident protocols
Schema/ontology definitions — to be created as schema/protocol_schema.json and schema/ontology.json
Gold standard annotations — to be created as evaluation/gold/*.json
Extraction pipeline — to be created as Python modules under src/
Verification

Run protocol section detector on IRPG-2025, compare detected protocol boundaries against manually identified boundaries — target >80% F1
Extract 5-8 gold-standard protocols end-to-end, compute node precision/recall and edge precision/recall against manual annotations
Run structural validator on all extracted graphs — zero structural errors allowed
Human reviewers score extracted protocols — target average ≥3.5/5 on correctness and completeness
Demonstrate at least one cross-protocol KG query working correctly (e.g., "which protocols require PPE?")
Decisions

LLM-first extraction over traditional CV pipeline — modern multimodal LLMs handle flowchart understanding well enough for a thesis prototype; building a box/arrow detector from scratch is a separate thesis
NetworkX over Neo4j — thesis prototype doesn't need a production graph DB; JSON export covers persistence
Few-shot prompting over fine-tuning — with 5-8 gold examples, few-shot is sufficient and avoids the data scarcity problem
Scope: English only, native digital PDFs only, IRPG-2025 + 2-3 similar emergency protocol documents
Out of scope: scanned PDFs, OCR pipeline, handwritten annotations, real-time processing, production deployment
Further Considerations

LLM choice: Claude (Anthropic) vs GPT-4o (OpenAI) for extraction — recommend testing both on 2-3 protocols and comparing accuracy before committing. Claude tends to follow structured output schemas more reliably; GPT-4o has stronger vision capabilities
Cost management: Processing every page with a vision LLM is expensive (~$0.01-0.05/page). The section detector in step 5 is important to avoid processing non-protocol pages. Estimate ~$5-15 per full document pass
Thesis framing: I'd recommend framing this as "Automated Extraction and Knowledge Graph Construction of Emergency Response Protocols from PDF Documents" — emphasizing the two-layer architecture (procedural flow + semantic knowledge) as the contribution, with evaluation demonstrating that the structured representation enables downstream tasks (querying, cross-protocol analysis) that plain text extraction cannot






Critique reply:
My take: the critique is mostly strong, and it improves the original plan in the right direction. It correctly pushes the project from a broad “platform architecture” into a thesis-sized extraction-and-evaluation pipeline. I agree especially with its emphasis on gold-standard annotation first, a small ontology, NetworkX/JSON over a graph database, and an LLM-first strategy for diagrams instead of building a custom box-arrow CV stack. Those are good scope controls.

Where I agree most strongly with the critique:

The original plan was a bit too “system design” heavy. For a thesis, the contribution should be something like: can protocol structure be extracted reliably enough to support graph construction and querying? That means annotation, extraction, validation, and evaluation should dominate the work, not storage infrastructure. Also, the point about using JSON + NetworkX first is right; graph databases are useful later, but not necessary to prove the research idea. PyMuPDF already gives you text, images, and block-level extraction, while tools like Docling and Marker already provide structured document outputs, so you do not need to spend thesis time reinventing the ingestion layer.

I also agree with the critique that the “three totally separate extractors” idea is too clean for real PDFs. In practice, protocol sections are often mixed-format. A better unit of processing is the protocol section, not the page and not the modality. For one section, you may want to pass: extracted text, table content, and one or more page images together into a structured extraction step. That is exactly where current document tools and multimodal models help: Docling’s document model includes hierarchy, layout, tables, provenance, and bounding boxes, while Marker can export markdown/JSON and even supports schema-based structured extraction in beta.

Where I would soften the critique a bit:

I do not think the original “page classifier” has to become a separate research problem. The critique is right that a learned classifier would add scope, but you do not need one at first. A rule-based detector using headings, numbering patterns, table presence, and visual cues is a valid thesis choice. PyMuPDF can expose blocks and text order, and document parsers like Docling already preserve hierarchy and layout, which makes heuristic routing much easier than the critique suggests. So I would reframe that point as: do not build a trained classifier unless the heuristics fail.

I also think the critique slightly overstates the need to name every library up front. It is true that the original discussion stayed too abstract, but in a thesis proposal you do not need a shopping list of ten tools. You need a small, defensible stack. For example: PyMuPDF or Docling for ingestion/layout, multimodal LLM for section extraction, structured outputs for JSON, NetworkX for the graph. That is concrete enough without overcommitting early. Both Anthropic and OpenAI now support structured-schema workflows, which strengthens this design choice.

The most important missing nuance in the critique is this:

The project should not start as “knowledge graph extraction.” It should start as protocol graph extraction, then optionally enrich into a knowledge graph. That is still the safest intellectual framing. The procedural layer is the hard, primary contribution. The semantic KG layer is valuable, but it should be presented as a second layer built on top of a correct protocol graph, not as the first problem. Otherwise the thesis risks drifting into ontology design and NER instead of solving the central challenge: preserving procedural branching.

So if I were refining the refined plan, I would make the stages:

Gold standard first
Annotate 5–8 protocols as decision/protocol graphs.

Section detection
Heuristic first, not ML first.

Section-level multimodal extraction
One section in, one structured protocol graph out.

Graph validation + evaluation
Node/edge precision-recall, structural checks, human scoring.

Light semantic enrichment
Only a few entity types: role, equipment, hazard, incident, outcome.

Demonstration
Show at least one useful downstream query or scenario-generation use case.

That keeps the thesis centered.

I especially like the critique’s suggestion to use few-shot structured extraction against your own gold annotations. That is probably the best research move here. Modern APIs support schema-constrained outputs, and current multimodal tooling can read images and PDFs directly, so a section-level extraction workflow is realistic without building a full custom vision pipeline.

One point I would change in the refined plan: I would be careful with the wording “LLM-first extraction over traditional CV pipeline — modern multimodal LLMs handle flowchart understanding well enough.” That is directionally right, but I would not promise too much. For a thesis, say instead: multimodal LLMs are used as the primary structured extractor, with evidence grounding and post-validation to catch failures. That wording is safer academically because it does not overclaim robustness.

Another point: the critique is right that cost and latency deserve mention, but I would keep that section small. The main architectural implication is simply: do not send every page as an image unless needed. Use section detection and native-text extraction first, then reserve image-based multimodal calls for diagram-heavy or mixed sections. That is enough for a thesis proposal; the exact dollar estimate should come from a pilot experiment, not from guesswork. OpenAI’s current pricing and vision docs support the general point that multimodal inputs have real cost, and Anthropic’s PDF support confirms document-level multimodal processing is available, so selective use matters.

So my final judgment is:

The critique is better than the original plan in terms of thesis scope.
But I would refine it one more step into this principle:

Thesis contribution = extract protocol graphs from PDFs reliably enough to support a small protocol-centric knowledge graph.

Not:
Build a full protocol KG platform.
That distinction matters a lot.



Claud's reply back:
Response to the Reply
Points I accept and will incorporate into the plan:

"Protocol graph first, KG second" as the thesis framing. This is the strongest point in the reply. The current plan already has this order (Phase 3 before Phase 4), but the framing matters — the thesis contribution should be stated as "can protocol structure be extracted reliably?" with the KG as an enrichment layer, not the headline. I'll adjust the plan's title and contribution statement accordingly.

Section-level multimodal extraction, not modality-level. The reply correctly points out that real protocol sections mix text, tables, and diagrams on the same page. Instead of routing to three separate extractors, the right unit is: pass the entire section (text + tables + images together) into one multimodal extraction call. This is simpler and avoids the integration/merge problem I flagged in the critique. I'll collapse the three extractors into one section-level extraction step.

Heuristic-first section detection, not ML-first. The reply is right that I slightly overframed the page classifier as "a research problem in itself." For a thesis, a rule-based detector using headings, numbering patterns, and layout cues from PyMuPDF is a valid and defensible choice. Only escalate to a learned classifier if heuristics fail. I'll reframe step 5-6 accordingly.

Softer LLM robustness claim. Changing from "LLMs handle flowcharts well enough" to "multimodal LLMs are used as the primary extractor, with evidence grounding and post-validation to catch failures" is a better academic framing. Small wording change, but it avoids overclaiming.

Selective multimodal calls to manage cost. Use native text extraction first, reserve image-based vision calls only for diagram-heavy or mixed sections. The cost estimate should come from a pilot experiment, not upfront guesswork. I agree — I'll remove the dollar estimates and replace with "run a pilot cost experiment in Phase 3."

Docling / Marker mention. The reply correctly names tools I didn't explore enough. Docling in particular provides hierarchy, layout, tables, provenance, and bounding boxes out of the box — which could replace or augment PyMuPDF for ingestion. Worth evaluating in Phase 2.

Points I push back on:

"Do not need a shopping list of ten tools." I disagree with the framing but not the substance. The plan doesn't need ten tools, but it does need to name the 3-4 key ones with justification. A thesis methodology section that says "we used document parsing and an LLM" is too vague to reproduce. Naming PyMuPDF (or Docling), a specific LLM API, and NetworkX is the minimum for reproducibility. The reply essentially agrees with this ("small, defensible stack") — so this is a non-disagreement.

"Light semantic enrichment — only a few entity types." The reply suggests: role, equipment, hazard, incident, outcome. I'd argue this is too light. If you only extract 5 entity types, the "knowledge graph" label becomes hard to justify — it's more like tagged metadata on a protocol graph. I'd keep the original plan's 8-10 node types and 10-12 relation types as the target, but clarify that extraction is LLM-based (not a custom NER model) and that completeness is evaluated but not expected to be perfect. That's the honest thesis position.

The reply's 6-stage outline is good but missing evaluation detail. The proposed stages (gold standard → section detection → extraction → validation → enrichment → demonstration) are a clean reordering, but the reply doesn't specify how evaluation works — it just says "node/edge precision-recall, structural checks, human scoring." Our plan already has more detail here (specific metrics, F1 targets, reviewer protocol), and I'd keep that.

Now let me update the plan to incorporate the accepted improvements:

Updated memory file plan.md

Here's the updated plan incorporating the reply's best points:

Updated Plan: Protocol Graph Extraction from PDFs (v2)
Thesis contribution: Can protocol structure be extracted from multi-format PDFs reliably enough to support a protocol-centric knowledge graph? The primary contribution is protocol graph extraction with evaluation; the KG is a secondary enrichment layer.

What changed from v1 based on the reply:

Collapsed three separate extractors into one section-level multimodal extraction step (text + tables + images for one section go into one LLM call)
Reordered: validation/evaluation now comes before KG enrichment — the protocol graph is the core deliverable
Section detection is explicitly heuristic-first (headings, numbering, imperative verbs, layout cues), not ML
Added Docling as a candidate alongside PyMuPDF for ingestion (evaluate both early)
Softer LLM framing: "primary structured extractor with post-validation," not "handles it well enough"
Cost: run a pilot cost experiment in Phase 3 instead of guessing upfront
KG enrichment kept at 8-10 node types / 10-12 relation types (not reduced to 5) — otherwise the "knowledge graph" label is hard to justify academically
Phases

Phase 1: Gold Standard & Schema (blocks everything)

Manually annotate 5-8 protocols from IRPG-2025 as decision/protocol graph JSON — mix of text-step, table, and diagram protocols
Define protocol graph JSON schema: node types (start, action, decision, end), edge types (next, true_branch, false_branch), metadata (protocol_type enum, confidence, cross_references, evidence with page/bbox/text)
Define KG ontology: 8-10 node types (Protocol, Step, Decision, Action, Role, Equipment, Hazard, Incident, Outcome, SourcePage), 10-12 relation types (HAS_STEP, NEXT, BRANCHES_TO, REQUIRES, PERFORMED_BY, APPLIES_TO, USES, LEADS_TO, MENTIONED_IN, DERIVED_FROM)
Phase 2: PDF Ingestion & Section Detection (parallel with Phase 1)
4. Evaluate PyMuPDF vs Docling for text/layout/table extraction on IRPG-2025 — Docling provides hierarchy, provenance, and bounding boxes natively; PyMuPDF is simpler. Pick one
5. Build heuristic-based protocol section detector: heading keywords, numbering patterns, ToC parsing, imperative verb detection, layout cues. Output: {title, start_page, end_page, format_hint} per protocol. Only escalate to a learned classifier if heuristics fail
6. For each section, tag as text_steps | table | diagram | mixed using simple rules — this is a routing hint for the extraction prompt, not a hard classifier

Phase 3: Section-Level Multimodal Extraction (depends on 1+2)
7. For each detected protocol section, feed the full section context into one LLM call: extracted text + table content + page image(s) for diagram/mixed sections. Use structured output schemas (both Anthropic and OpenAI support this). Include 2-3 few-shot examples from gold standard. Output: one protocol graph JSON per section
8. Use native text extraction first; only add page images for sections tagged diagram or mixed — this controls cost
9. Run a pilot cost/accuracy experiment on 3-5 sections: compare Claude vs GPT-4o on accuracy and cost. Pick one as primary
10. Handle cross-page protocols by concatenating text / providing sequential images with explicit "this is one protocol" instruction

Phase 4: Validation & Evaluation (depends on 3 + gold standard from 1)
11. Structural validation (automated): start node exists, decision nodes have ≥2 outgoing edges, no orphan nodes, no dangling edge references
12. Content accuracy (against gold standard): node-level precision/recall, edge-level precision/recall, protocol boundary detection F1 — target >80% F1
13. Confidence scoring: based on extraction method (text > table > diagram) + LLM self-reported uncertainty
14. Human evaluation: 1-2 reviewers score 10-15 extracted protocols on correctness (1-5) and completeness (1-5) — target ≥3.5 average

Phase 5: Semantic KG Enrichment (depends on 3, lighter than protocol graph work)
15. Second LLM pass over extracted protocols to identify semantic entities (roles, equipment, hazards, conditions, incident types, outcomes) and typed relations using the ontology from step 3
16. Entity linking across protocols — merge duplicate entities (e.g., "Paramedic" across protocols → one node with edges to each)
17. Store in NetworkX + export to JSON/JSON-LD

Phase 6: Demonstration (depends on 4+5)
18. Visualize extracted protocol graphs alongside source PDF pages (Graphviz/pyvis)
19. Demonstrate at least one downstream KG query: "which protocols require PPE?", "what actions follow smoke detection?", or similar
20. Optionally: show a simple scenario-generation use case from protocol branches

Verification

Section detector on IRPG-2025: compare against manual protocol boundaries → >80% F1
End-to-end extraction on 5-8 gold protocols: node P/R, edge P/R
Structural validator: zero structural errors on all extractions
Human reviewers: ≥3.5/5 on correctness and completeness
At least one working cross-protocol KG query
Scope

In: English, native digital PDFs, IRPG-2025 + 2-3 similar emergency protocol documents, heuristic section detection, LLM-based extraction, small KG
Out: scanned PDFs, OCR, handwritten annotations, trained classifiers (unless heuristics fail), graph databases, production deployment
Does this feel right for what you want to build? The main open question is whether you want me to flesh out the schema/ontology design (Phase 1) next, or whether you want to adjust the phases further.