# Literature Review Research — 2026-04-12

## Overview

Reviewed the existing literature to avoid reinventing the wheel. The research covers three connected strands: PDF information extraction, process/protocol extraction from text, and knowledge-graph construction. The main finding is that truly end-to-end work going from PDF directly to a faithful protocol graph with branching logic is still rare — most papers solve one layer well, then hand off to another.

---

## Papers Already Identified (8 Papers)

### 1. Atagong et al., 2025 — Review of Information Extraction from PDFs

Best entry point for the document side. Reports that recent systems commonly combine OCR and PDF parsers in pre-processing, then use rule-based, statistical, or neural IE methods, and finally store outputs in JSON, XML, databases, or triplet/graph structures. Names concrete front-end technologies: Tesseract, PyMuPDF, PDFX, XPDF, PDFMiner. Notes growing use of GPT-3.5 and Mistral in the processing stage. Found 30 eligible studies from 2017–2025 and described a generic pre-processing → processing → storage pipeline.

### 2. Van Woensel & Motie, 2024 — Systematic Review on Process Extraction

Best entry point for the protocol/process side. Defines automated process extraction as transforming text into structured processes. Distinguishes NLP from process generation. Shows the field has shifted from rule-based systems toward BERT, LSTM, and other DL models. Catalogues common NLP tools (spaCy, NLTK, Stanford CoreNLP, OpenNLP, Stanza) and target outputs (BPMN, declarative models, decision models).

### 3. Grathwol, van der Aa, and López, 2024 — Automating Pathway Extraction from Clinical Guidelines

One of the closest papers to the thesis direction. Explicitly about extracting pathway/process information from clinical guidelines. Proposes the CGPET conceptual model capturing goals, indications/contra-indications, plans, classification rules, and decision rules. Compared BioLinkBERT, XLM-RoBERTa, GPT-4-0125 Preview, Mixtral 8x7B Instruct, and GoLLIE-34B. Conclusion: task is still very hard — relation extraction looked more promising but entity extraction was not strong enough for production use.

### 4. He, Tang, and Wang, EMNLP 2024 — Medical Decision Rule Extraction from Text

One of the strongest papers on explicit branching logic extraction. Models medical rules as binary trees with condition and decision nodes, then uses generative models to produce linearised trees in natural language, augmented natural language, or JSON and parse them back into tree structure. Best system reached 67% tree accuracy and 78% path F1 on a Chinese benchmark. Notes that entity normalisation to UMLS is not handled yet.

### 5. Cui et al. / Yang et al., 2024 — Review on Healthcare Knowledge Graphs

Best high-level source for the graph side in healthcare. Lays out a clean pipeline: develop schema → collect data → extract triples → normalise entities/relations → infer missing links → update and validate. Separates the document/protocol extraction problem from the later graph-population problem.

### 6. Hofer et al., 2024 — Construction of Knowledge Graphs: Current State and Challenges

Emphasises that KG construction needs not just extraction, but also metadata management, provenance, ontology development, entity resolution, knowledge completion, and quality assurance. Recommends fact-level provenance (original text paragraph, confidence score, source metadata) so extracted facts can be debugged and updated later.

### 7. Allocca et al. — Knowledge Graph Construction for Health, Lifestyle and Recommendations

Example of how extracted guideline knowledge is operationalised. Uses RML to map heterogeneous data to RDF, converts FHIR to RDF triples, applies RDFox with OWL Functional Syntax rules. The rule layer is interpreted from validated guidelines. Less about PDF extraction, more about the reasoning layer after extraction.

### 8. Nundloll et al., 2022 — Historical Text to Linked Data Model

Full document-to-linked-data workflow combining ML, NLP, and Semantic Web techniques. Good example of end-to-end architecture from text to queryable linked-data model.

---

## New Papers Found (10 Papers)

### 9. Neuberger et al., 2024 — *A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text using LLMs* (arXiv:2407.18540)

**MOST IMPORTANT MISSING PAPER.** Same group as Grathwol/van der Aa/López (van der Aa is co-author on both). Systematically tests 8 LLMs for extracting activities, actors, and relations from textual process descriptions. Develops a prompting strategy that outperforms state-of-the-art ML by up to 8% F1. Key findings:

- Number of few-shot examples, specificity of definitions, and rigour of format instructions are the three most impactful prompt design factors
- Code, prompts, and data are publicly available

**Relevance:** Closest methodological comparison to this thesis. The LLM extraction with few-shot prompting and JSON schema is essentially the same approach, applied to emergency protocols instead of generic process descriptions. Can benchmark against them.

### 10. Gupta et al., 2024 — *Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs* (arXiv:2501.13984)

Creates faithful graph representations of NCCN Cancer CPGs using automated extraction + LLM-based node classification (80.86% zero-shot, 88.47% few-shot). Also builds subgraph-based QA to mitigate hallucination.

**Relevance:** Closest existing system to what this thesis builds — graphical clinical guidelines → knowledge graph → structured queries — just in oncology instead of emergency protocols.

### 11. Ye et al., 2024 — *Beyond End-to-End VLMs: Leveraging Intermediate Text Representations for Superior Flowchart Understanding* (arXiv:2412.16420)

Proposes **TextFlow**: a two-stage pipeline (Vision Textualiser → Textual Reasoner) that converts flowchart images to intermediate text representations (Graphviz, Mermaid, PlantUML) before reasoning. Key finding: **intermediate structured text beats end-to-end VLM** for controllability, explainability, and modularity. SOTA on FlowVQA and FlowLearn benchmarks. Code is public.

**Relevance:** Directly addresses the multimodal extraction path. The current GPT-4o vision extraction is essentially their "end-to-end VLM" approach, which they show is suboptimal. Their two-stage approach (image → Mermaid/Graphviz → reason) could improve Triage-style diagram protocols.

### 12. Omasa et al., 2025 — *Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding* (arXiv:2505.07864)

Seven-stage pipeline: arrow-aware node/endpoint detection → OCR → structured prompt for VLMs. Raises flowchart accuracy from 80% to 89%. Next-step queries reach 100% accuracy. Key insight: explicitly encoding arrow direction eliminates the most common VLM failure mode.

**Relevance:** Edge F1 of 0.444 on Triage multimodal extraction could benefit from this — arrow direction encoding addresses exactly the "edge" problem.

### 13. Castillo & Mukkamala, 2026 — *Guardian Parser Pack: LLM-based Schema-Guided Extraction and Validation* (arXiv:2604.06571)

Schema-first extraction pipeline for heterogeneous documents: multi-engine PDF extraction with OCR fallback → rule-based source identification → schema-first harmonisation and validation → LLM-assisted extraction with validator-guided repair. LLM pathway achieves F1=0.8664 vs. deterministic baseline 0.2578.

**Relevance:** Very similar architecture (schema-first, validation loop, LLM-assisted). Their "validator-guided repair" concept maps to the two-stage validator. Good comparative reference.

### 14. Grohs et al., 2023 — *Large Language Models can accomplish Business Process Management Tasks* (arXiv:2307.09923, BPM 2023)

Shows LLMs can mine both imperative and declarative process models from text descriptions, performing comparably to or better than existing solutions without extensive prompt engineering. Important early evidence that LLMs can handle process extraction directly.

### 15. Docling — Auer et al., 2024 (arXiv:2408.09869)

IBM's open-source PDF conversion tool using DocLayNet for layout analysis and TableFormer for table structure recognition. MIT-licensed, runs on commodity hardware. The Refined Plan mentions Docling by name as a tool option — this is the technical report to cite.

### 16. Lundin et al., 2025 — *From Guidelines to Guarantees: A Graph-Based Evaluation Harness for Domain-Specific LLMs* (arXiv:2508.20810)

Transforms WHO IMCI clinical guidelines into a queryable knowledge graph, then generates evaluation queries via graph traversal. Three guarantees: complete coverage, contamination resistance, and inherited validity.

**Relevance:** Relevant for evaluation methodology — shows how to evaluate LLM performance on guideline knowledge using the graph structure itself.

### 17. Liu et al., 2025 — LLM-based CINV Knowledge Graph for Breast Cancer

Uses the Qwen model under the CRISPE framework for NER and relation extraction from 47 studies. Extracted 273 entities and 289 relations; after expert validation, 238 entities and 242 relations retained. NER F1 = 82.97, RE F1 = 85.54. Visualised in Neo4j.

### 18. ter Horst et al., 2023 — Ontology-Driven KG Population for Pre-Clinical Studies

Uses ontology-guided model-complete text comprehension and conditional random fields in a hierarchical architecture. Describes outcomes by up to 103 parameters. Shows how domain ontology creates deep relational data-structures from publications.

---

## Consolidated Pipeline the Literature Supports

1. **Ingest and pre-process the PDF** — PyMuPDF, PDFMiner, PDFX, XPDF for born-digital; Tesseract for scanned; Docling, LayoutLM, Donut for visually rich documents.

2. **Run sentence- and token-level IE** — tokenisation, POS tagging, NER, relation extraction, event extraction, sentence classification. Tools: spaCy, NLTK, CoreNLP, CRFs, SVMs, LSTMs, CNNs, BERT-family, SciBERT, XLM-RoBERTa, GPT-4, Mixtral, GoLLIE.

3. **Reconstruct process or decision structure** — hardest part. Needs an explicit conceptual schema (CGPET-style). Decision-tree extraction with structured generation is more promising than flat extraction for branching logic.

4. **Normalise into ontology-backed concepts** — map entities/relations to controlled vocabularies or ontologies. Healthcare KG review explicitly includes normalisation.

5. **Store as graph plus rule layer** — RDF/RML/OWL/SPARQL for reasoning. Protocol behaviour typically lives in a decision tree, process graph, or rule layer on top of the KG.

6. **Preserve provenance and quality checks** — fact-level metadata (source paragraph, confidence score, versioning, provenance).

---

## Gap Analysis: What's Missing From the Review

### Critical Gap: Flowchart Understanding

The current literature review has **no coverage** of the flowchart/diagram understanding sub-field, even though the pipeline uses multimodal extraction (GPT-4o vision) for diagram protocols and reports that it's the hardest case (Triage Edge F1 = 0.083 text-only → 0.444 multimodal). Ye 2024 and Omasa 2025 provide evidence-backed techniques that could directly improve this weakest link.

### Missing Strand: Schema-Guided Extraction

Castillo 2026 validates the schema-first approach with quantitative results.

---

## Where Each Paper Fits in the Pipeline

| Pipeline Layer | Original papers | New papers to add |
|---|---|---|
| **PDF ingestion** | Atagong 2025 | **Docling** (Auer 2024) |
| **Process/protocol extraction** | Van Woensel & Motie 2024; Grathwol 2024 | **Neuberger 2024**; **Grohs 2023** |
| **Decision/branching logic** | He/Tang/Wang EMNLP 2024 | — (solid) |
| **Clinical guideline → graph** | Grathwol 2024 | **Gupta 2024**; **Lundin 2025** |
| **Flowchart/diagram understanding** | *(missing entirely)* | **Ye 2024**; **Omasa 2025** |
| **Schema-guided extraction** | *(missing entirely)* | **Castillo 2026** |
| **KG construction** | Cui/Yang 2024; Hofer 2024; Allocca et al. | — (solid) |
| **Provenance & QA** | Hofer 2024; Nundloll 2022 | — (solid) |

---

## What the Literature Confirms About the Current Architecture

1. **Layered pipeline design is well-supported.** Every paper that works well uses modular stages, not monolithic end-to-end approaches.
2. **Few-shot prompting with JSON schema is the consensus approach.** Neuberger 2024, Gupta 2024, and Castillo 2026 all converge on this.
3. **Two-stage validator (structural + semantic) is novel in the protocol extraction space.** Castillo 2026 has a similar concept ("validator-guided repair") but the current design is more sophisticated with 8 structural checks + 8 semantic checks.
4. **Confidence-gated batching has no direct precedent** in the papers found — most papers assume all sections are extractable. This is a genuine contribution.
5. **Intermediate text representation for diagrams** (Ye 2024) is a stronger approach than direct VLM extraction — worth acknowledging as future work or trying.

---

## Comparison Matrix

| Paper | Year | Input | Domain | PDF handling | Extraction target | Output format | Models/tools | Evaluation | Replication |
|---|---|---|---|---|---|---|---|---|---|
| Neuberger et al. | 2024 | Process descriptions | Generic BPM | N/A (text) | Activities, actors, relations | Process model | 8 LLMs, prompting strategy | F1 on 3 datasets | Code + data public |
| Gupta et al. | 2024 | NCCN guidelines | Oncology | Manual/semi-auto | Nodes, relationships, pathways | Graph + QA | LLMs (zero/few-shot) | 80.9–88.5% node classification | Partial |
| Grathwol et al. | 2024 | Clinical guidelines | Healthcare | N/A (text) | CGPET entities + relations | Conceptual model | BioLinkBERT, GPT-4, Mixtral, GoLLIE | Entity/relation F1 | Limited |
| He/Tang/Wang | 2024 | Medical text | Healthcare (CN) | N/A (text) | Decision trees (binary) | Linearised trees → JSON | Generative models | 67% tree acc, 78% path F1 | Benchmark |
| Ye et al. | 2024 | Flowchart images | Generic | Image input | Graph structure + QA | Graphviz/Mermaid/PlantUML | VLMs + LLMs (2-stage) | FlowVQA, FlowLearn | Code public |
| Omasa et al. | 2025 | Flowchart images | Generic | Image input | Nodes, edges, direction | Structured prompt | VLMs + arrow detection | 89% accuracy | Limited |
| Castillo et al. | 2026 | Heterogeneous docs | Missing persons | Multi-engine PDF + OCR | Schema-compliant records | Unified JSON | LLM + rule-based | F1=0.8664 (LLM) | Architecture |
| Lundin et al. | 2025 | WHO IMCI guidelines | Clinical | N/A | KG + eval queries | Graph traversal | 5 LLMs evaluated | Coverage, contamination resistance | Framework |
| Docling (IBM) | 2024 | Any PDF | Generic | DocLayNet + TableFormer | Structured document | Markdown/JSON | DocLayNet, TableFormer | Layout/table benchmarks | MIT, open source |
| **This pipeline** | 2025 | Emergency PDFs | IRPG, VIC Bushfire | PyMuPDF + TOC | Protocol graphs | JSON (node/edge) | GPT-4o, GPT-4o-mini | Node F1=0.833, Edge F1=0.628 | — |

---

## Key Takeaway

The research opportunity is real: the gap is not "can we extract facts from PDFs?" but "can we preserve branching, ordering, and context dependence reliably enough for protocols?" The strongest literature-backed positioning is: **a layered pipeline for protocol reconstruction from PDFs, where PDF parsing, process/decision extraction, and KG population are treated as separate modules, with explicit provenance and a dedicated rule/graph layer for branching logic.**
