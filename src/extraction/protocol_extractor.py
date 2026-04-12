"""
Protocol Extraction Module

Uses OpenAI GPT-4o to extract structured protocol graphs from PDF section text.
Supports few-shot prompting with gold standard examples.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.ingestion.pdf_ingest import DocumentContent, render_page_image
from src.extraction.section_detector import ProtocolSection


def _safe_json_loads(raw: str) -> dict:
    """Parse JSON, fixing common LLM output issues (invalid escapes, etc.)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Fix invalid unicode escapes: replace \uXXXX where XXXX isn't valid hex
    fixed = re.sub(
        r'\\u(?![0-9a-fA-F]{4})[^"\\]*',
        lambda m: m.group(0).replace('\\u', '\\\\u'),
        raw,
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract first { ... } block
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        return json.loads(raw[brace_start:brace_end + 1])

    raise json.JSONDecodeError("Could not parse LLM JSON output", raw, 0)


SYSTEM_PROMPT = """\
You are an expert at extracting structured protocol graphs from emergency response documents.

Given the text of a protocol section from a PDF, you must produce a JSON object representing the protocol as a directed graph.

## Output Schema

{
  "protocol_id": "<snake_case_id>",
  "title": "<protocol title>",
  "source_pdf": "<filename>",
  "pages": [<1-based page numbers>],
  "protocol_type": "<sequential | decision_tree | table | hybrid>",
  "extraction_confidence": <0.0 to 1.0>,
  "cross_references": ["<any references to other protocols or pages>"],
  "nodes": [
    {
      "id": "n1",
      "type": "<start | end | action | decision>",
      "text": "<node content>",
      "evidence": {
        "page": <page number>,
        "source_text": "<original text from document>"
      }
    }
  ],
  "edges": [
    {
      "from": "n1",
      "to": "n2",
      "condition": "<optional: branch condition like 'Yes', 'No', or a descriptive condition>"
    }
  ]
}

## Rules

### Structure
1. Every protocol must have exactly one "start" node and at least one "end" node.
2. Node IDs must be sequential: n1, n2, n3, etc.
3. No orphan nodes — every node must be connected via at least one edge.
4. Every node must have evidence linking back to the source text and page number.

### Node Types
5. "action" nodes represent imperative steps (things to do).
6. "decision" nodes represent questions or conditional checks (yes/no, if/then, conditional branching).
7. Use protocol_type "sequential" for linear step lists, "decision_tree" for branching logic, "table" for table-based protocols, "hybrid" for mixed formats.

### Edge & Branching Rules (CRITICAL — pay close attention)
8. Every "decision" node MUST have at least 2 outgoing edges, each with a DIFFERENT condition label.
9. For every if/then/else in the source text, create exactly ONE decision node with one edge per branch. Label each edge with the exact condition text (e.g., "Yes (breathing)", "No (not breathing)").
10. When the source says "if X, do A; otherwise do B", model as: decision node "X?" → edge "Yes" → action A, edge "No" → action B.
11. When a decision has more than 2 outcomes, create one edge per outcome with a descriptive condition for each.
12. After all branches of a decision are resolved, reconnect them to the next shared step or to "end" nodes. Do NOT leave branches dangling.
13. If the protocol has numbered steps, preserve the sequential order and connect them with edges.
14. Edge conditions should be clear and descriptive, matching the source text as closely as possible.

### Anti-Over-Extraction
15. Distinguish between **sequential steps** and **hierarchical details**:
    - Sequential steps (numbered items, or distinct actions in order): create SEPARATE nodes.
    - Hierarchical sub-bullets (details, examples, or attributes under a heading): CONSOLIDATE into ONE node whose text is the heading with key details. Sub-bullets are NOT separate steps.
    - Example: "Identify Hazards • Gather Information • Scout the Fire" → ONE action node "Identify Hazards: Gather Information, Scout the Fire".
16. Do NOT invent steps, intermediate nodes, or conditions that are not in the source text. Only extract what is explicitly stated.
17. Prefer fewer, more faithful nodes over more granular decomposition. Each node should correspond to a distinct step or decision in the source.
18. Observation/assessment categories (e.g., "Skin Color: Normal/Pale/Bluish") where you record a finding but the flow does NOT branch based on the result should be modeled as ACTION nodes, not decision nodes. Only use "decision" when the protocol flow actually diverges.

### Metadata
19. If the protocol references other protocols or pages, include them in cross_references.
20. Set extraction_confidence between 0.0 and 1.0 based on how clear and unambiguous the protocol is.
21. Return ONLY valid JSON. No markdown, no explanation, no code fences.
"""


TABLE_EXTRACTION_PROMPT = """\
You are an expert at extracting structured protocol graphs from emergency response documents \
that are formatted as TABLES.

Given a table-based protocol section, convert it into a JSON protocol graph.

## How to interpret table protocols

- Each ROW typically represents a condition-action pair, a checklist item, or a step.
- COLUMN HEADERS often indicate categories (e.g., "Condition", "Action", "Priority", "Notes").
- If the table has a condition/criteria column and an action column, model each row as: \
decision node (condition) → action node (action).
- If the table is a sequential checklist (no conditions), model rows as sequential action nodes \
connected by edges.
- If the table groups rows by category/severity/priority, use the grouping as decision branches.
- Preserve the exact cell text as node text — do not paraphrase.

## Output Schema

Same as the standard protocol schema: nodes with id/type/text/evidence, edges with from/to/condition.

## Rules

1. Every protocol must have exactly one "start" node and at least one "end" node.
2. Every "decision" node must have at least 2 outgoing edges with different conditions.
3. Node IDs must be sequential: n1, n2, n3, etc.
4. Every node must have evidence linking to the source text and page number.
5. Do NOT invent steps not in the source table. Only extract what is explicitly present.
6. Set extraction_confidence based on how clearly the table maps to a process flow.
7. Return ONLY valid JSON. No markdown, no explanation, no code fences.
"""


def _load_few_shot_examples() -> list[dict]:
    """Load gold standard files to use as few-shot examples."""
    gold_dir = Path(__file__).parent.parent.parent / "evaluation" / "gold"
    # Use CPR (decision_tree), Burn Injuries (sequential), and
    # Triage (complex decision_tree with 5 decision nodes) as examples
    # to demonstrate both formats and complex branching
    example_files = ["cpr_01.json", "burn_injuries_01.json", "triage_01.json"]
    examples = []
    for fname in example_files:
        fpath = gold_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                examples.append(json.load(f))
    return examples


def _build_few_shot_messages(examples: list[dict]) -> list[dict]:
    """Build few-shot user/assistant message pairs from gold examples."""
    messages = []

    # Example source texts paired with each gold annotation
    example_texts = {
        "cpr_01": (
            "CPR\n"
            "1. Scene Safety: Look for any dangers or hazards.\n"
            "2. Determine Responsiveness: Tap on the patient's shoulders and shout, "
            "\"Are you OK?\" Look for chest rise and fall. If the patient is not breathing, "
            "continue with steps 3 and 4. If the patient is breathing and no spinal injury "
            "is suspected, place patient on their side. Continue to monitor breathing.\n"
            "3. Call for Help: Activate emergency response. If possible, obtain an automated "
            "external defibrillator (AED).\n"
            "4. Chest Compressions: Place the heel of one hand on the center of the patient's "
            "chest. Place the other hand over the first and interlock the fingers. Perform "
            "compressions at a rate of 100 to 120 per minute, compressing the patient's chest "
            "at least two inches. Push hard and fast. Perform 30 compressions.\n"
            "5. Airway: Open the patient's airway by tilting the head back and lifting the chin. "
            "If trauma is suspected and you are trained, use the jaw thrust.\n"
            "6. Breathing: If possible, use a barrier device. Place the barrier device over the "
            "patient's nose and mouth. Pinch the patient's nose and give 2 breaths, making the "
            "chest rise. If no barrier device is available, perform continuous compressions with "
            "no breaks or perform mouth-to-mouth.\n"
            "7. Continue CPR: Continue alternating 30 compressions and 2 breaths. If a second "
            "rescuer arrives, one person can perform ventilations and one person can perform "
            "compressions. Maintain the same 30:2 ratio.\n"
            "8. AED: If an AED arrives, turn the AED on and follow the instructions provided."
        ),
        "burn_injuries_01": (
            "Burn Injuries\n\n"
            "INITIATE MEDICAL EVACUATION IMMEDIATELY.\n\n"
            "• Remove person from heat source while looking for signs of a burned airway "
            "(e.g., singed facial or nasal hairs, soot or burns around or in nose, mouth, "
            "black sooty sputum, etc.).\n"
            "• Apply cool, clear water over burned area. Do not soak person or use cold water "
            "and ice packs, as this may cause hypothermia.\n"
            "• Examine for other injuries.\n"
            "- Provide basic first aid.\n"
            "- Monitor airway, breathing, circulation (ABCs).\n"
            "- Treat for shock by keeping person warm, feet elevated.\n"
            "- Provide oxygen, if available and trained to administer.\n"
            "• Assess degree of burn and area affected.\n"
            "- First Degree (Superficial) - Red, mild to moderate pain.\n"
            "- Second Degree (Partial Thickness) - Skin may be red and raw, blistered, "
            "swollen, painful to very painful.\n"
            "- Third Degree (Full Thickness) - Whitish, charred, or translucent, no pin "
            "prick sensation in burned area.\n"
            "- Rule of Palms: Patient's palm = 1% of their body surface.\n"
            "• Cut away only burned clothing. Do not cut away clothing stuck to burned skin. "
            "Remove jewelry near injured area.\n"
            "• Loosely wrap burned area with clean, dry dressing, and moisten with clean water, "
            "and apply dry dressing on top.\n"
            "• For severe burns or burns covering large area of the body:\n"
            "- Loosely cover burned area with clean, dry dressing if available.\n"
            "- If dressings are unavailable, mylar blankets, plastic wrap, or clean blankets "
            "or clothing can be used.\n"
            "• Monitor ABCs.\n"
            "• Keep patient warm and dry. DO NOT use wet dressings or blankets.\n"
            "• Avoid hypothermia and overheating."
        ),
        "triage_01": (
            "Multi-Casualty Triage System\n\n"
            "Can the patient walk?\n"
            "Yes: GREEN (Minor) — All walking wounded; treatment can be delayed.\n\n"
            "No (cannot walk): Is the patient breathing?\n"
            "No (not breathing): Reposition airway.\n"
            "  Breathing after repositioning airway?\n"
            "  No: BLACK (Deceased/Dying) — Dead or with injuries not compatible with life. "
            "No respirations after repositioning airway.\n"
            "  Yes: continue to respirations check.\n\n"
            "Yes (breathing): Are respirations more than 30/minute?\n"
            "Yes (respirations >30/min): RED (Immediate) — Serious, life-threatening injury. "
            "Breathing but unconscious; respirations more than 30/minute.\n\n"
            "No (respirations <30/min): Is radial pulse present and capillary refill less than 2 seconds?\n"
            "No (pulse absent or refill >2s): RED (Immediate) — Radial pulse absent, "
            "capillary refill more than 2 seconds.\n\n"
            "Yes (pulse present, refill <2s): Can the patient follow simple commands?\n"
            "No (cannot follow commands): RED (Immediate) — Cannot follow simple commands.\n"
            "Yes (can follow commands): YELLOW (Delayed) — Respirations less than 30/minute. "
            "Radial pulse present, capillary refill less than 2 seconds. "
            "And can follow simple commands. Treatment and transport delayed."
        ),
    }

    for example in examples:
        pid = example["protocol_id"]
        if pid in example_texts:
            messages.append({
                "role": "user",
                "content": (
                    f"Extract the protocol graph from this section.\n"
                    f"Source PDF: IRPG-2025.pdf\n"
                    f"Pages: {example['pages']}\n\n"
                    f"--- SECTION TEXT ---\n{example_texts[pid]}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(example, indent=2),
            })

    return messages


def extract_protocol(
    section: ProtocolSection,
    doc: DocumentContent,
    model: str = "gpt-4o",
    use_few_shot: bool = True,
    toc_context: Optional[str] = None,
) -> dict:
    """Extract a structured protocol graph from a detected section.

    Args:
        section: Detected protocol section with page boundaries.
        doc: Full document content from PDF ingestion.
        model: OpenAI model name.
        use_few_shot: Whether to include gold standard examples in the prompt.

    Returns:
        Parsed protocol graph dictionary.
    """
    client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Gather section text
    section_text = _gather_section_text(section, doc)

    # Build messages — use table-specific prompt for table format
    system_prompt = TABLE_EXTRACTION_PROMPT if section.format_hint == "table" else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if use_few_shot:
        examples = _load_few_shot_examples()
        messages.extend(_build_few_shot_messages(examples))

    context_line = f"\n{toc_context}\n" if toc_context else ""

    messages.append({
        "role": "user",
        "content": (
            f"Extract the protocol graph from this section.\n"
            f"Source PDF: {doc.filename}\n"
            f"Pages: {list(range(section.start_page, section.end_page + 1))}\n"
            f"Section title: {section.title}\n"
            f"Format hint: {section.format_hint}\n"
            f"{context_line}\n"
            f"--- SECTION TEXT ---\n{section_text}"
        ),
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = _safe_json_loads(raw)

    # Attach usage metadata
    result["_extraction_meta"] = {
        "model": model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "few_shot": use_few_shot,
    }

    return result


def extract_protocol_from_text(
    title: str,
    section_text: str,
    pages: list[int],
    source_pdf: str,
    format_hint: str = "unknown",
    model: str = "gpt-4o",
    use_few_shot: bool = True,
) -> dict:
    """Extract a protocol graph from raw text (without needing full doc/section objects).

    Useful for testing and experimentation.
    """
    client = OpenAI()

    system_prompt = TABLE_EXTRACTION_PROMPT if format_hint == "table" else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if use_few_shot:
        examples = _load_few_shot_examples()
        messages.extend(_build_few_shot_messages(examples))

    messages.append({
        "role": "user",
        "content": (
            f"Extract the protocol graph from this section.\n"
            f"Source PDF: {source_pdf}\n"
            f"Pages: {pages}\n"
            f"Section title: {title}\n"
            f"Format hint: {format_hint}\n\n"
            f"--- SECTION TEXT ---\n{section_text}"
        ),
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = _safe_json_loads(raw)

    result["_extraction_meta"] = {
        "model": model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "few_shot": use_few_shot,
    }

    return result


def extract_protocol_multimodal(
    section: ProtocolSection,
    doc: DocumentContent,
    pdf_path: str | Path,
    model: str = "gpt-4o",
    use_few_shot: bool = True,
    dpi: int = 200,
    toc_context: Optional[str] = None,
) -> dict:
    """Extract a protocol graph using both text and page images.

    Sends page images alongside text to handle diagram-heavy protocols
    where decision logic is conveyed visually (flowcharts, arrows, etc.).

    For decision_tree and hybrid formats, uses a TextFlow intermediate
    representation (image → Mermaid text → protocol JSON) to better
    capture arrow directions and branching logic (Ye et al. 2024).
    """
    client = OpenAI()

    section_text = _gather_section_text(section, doc)

    # Determine if we should use TextFlow intermediate representation
    use_textflow = section.format_hint in ("decision_tree", "hybrid")

    # Render page images
    page_images = []
    for pg_num in range(section.start_page, section.end_page + 1):
        img_bytes = render_page_image(pdf_path, pg_num, dpi=dpi)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        page_images.append(b64)

    mermaid_text = None
    if use_textflow and page_images:
        mermaid_text = _image_to_mermaid(client, page_images, model)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if use_few_shot:
        examples = _load_few_shot_examples()
        messages.extend(_build_few_shot_messages(examples))

    # Build multimodal user message with text + images
    context_line = f"\n{toc_context}\n" if toc_context else ""

    textflow_section = ""
    if mermaid_text:
        textflow_section = (
            f"\n\n--- FLOWCHART STRUCTURE (Mermaid) ---\n"
            f"The following Mermaid diagram was extracted from the page image(s). "
            f"Use this to understand the exact branching structure, arrow directions, "
            f"and decision flow. This is MORE RELIABLE than the raw text for understanding "
            f"the graph structure.\n\n{mermaid_text}\n"
        )

    content_parts: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Extract the protocol graph from this section.\n"
                f"Source PDF: {doc.filename}\n"
                f"Pages: {list(range(section.start_page, section.end_page + 1))}\n"
                f"Section title: {section.title}\n"
                f"Format hint: {section.format_hint}\n"
                f"{context_line}\n"
                f"IMPORTANT: The page image(s) below may contain flowcharts, diagrams, "
                f"or arrows that show decision logic NOT present in the text. "
                f"Use BOTH the text and the visual layout to extract the full protocol structure. "
                f"Pay special attention to arrows, branching paths, Yes/No labels, and color coding."
                f"{textflow_section}\n\n"
                f"--- SECTION TEXT ---\n{section_text}"
            ),
        }
    ]

    # Attach page images
    for b64 in page_images:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })

    messages.append({"role": "user", "content": content_parts})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = _safe_json_loads(raw)

    result["_extraction_meta"] = {
        "model": model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "few_shot": use_few_shot,
        "multimodal": True,
        "dpi": dpi,
        "textflow": mermaid_text is not None,
    }

    return result


def _image_to_mermaid(
    client: OpenAI,
    page_images_b64: list[str],
    model: str = "gpt-4o",
) -> Optional[str]:
    """Convert page images of a flowchart/diagram to Mermaid text representation.

    This implements the TextFlow intermediate representation approach
    (Ye et al. 2024): image → structured text → downstream extraction.
    The Mermaid text captures arrow directions and branching that the
    raw text extraction misses.
    """
    content_parts: list[dict] = [
        {
            "type": "text",
            "text": (
                "Describe this flowchart/decision diagram as a Mermaid flowchart.\n\n"
                "Rules:\n"
                "- Use `graph TD` (top-down) direction.\n"
                "- Identify every box/node and every arrow/connection.\n"
                "- For decision diamonds, use `{Decision text}` syntax.\n"
                "- For action boxes, use `[Action text]` syntax.\n"
                "- Label every arrow with its condition (Yes/No, or descriptive text).\n"
                "- Preserve the exact text in each box — do not paraphrase.\n"
                "- Include color-coded categories if visible (e.g., GREEN, RED, YELLOW, BLACK).\n\n"
                "Return ONLY the Mermaid code block. No explanation."
            ),
        }
    ]

    for b64 in page_images_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_parts}],
            temperature=0.0,
        )
        mermaid = response.choices[0].message.content
        # Strip markdown code fences if present
        if "```mermaid" in mermaid:
            mermaid = mermaid.split("```mermaid", 1)[1]
            mermaid = mermaid.split("```", 1)[0]
        elif "```" in mermaid:
            mermaid = mermaid.split("```", 1)[1]
            mermaid = mermaid.split("```", 1)[0]
        return mermaid.strip()
    except Exception:
        return None


def _gather_section_text(
    section: ProtocolSection, doc: DocumentContent
) -> str:
    """Concatenate text from all pages in a section."""
    parts = []
    for pg_num in range(section.start_page, section.end_page + 1):
        if pg_num < 1 or pg_num > len(doc.pages):
            continue
        page = doc.pages[pg_num - 1]
        parts.append(f"[Page {pg_num}]")
        parts.append(page.full_text)

        # Include table content if present
        for i, table in enumerate(page.tables):
            parts.append(f"\n[Table on page {pg_num}]")
            for row in table.cells:
                parts.append(" | ".join(cell for cell in row))
    return "\n".join(parts)


EDGE_REVIEW_PROMPT = """\
You are a quality reviewer for protocol graph extraction. Given a protocol graph JSON and the \
original source text, review the EDGES for correctness and completeness.

Check the following:
1. Every "decision" node must have at least 2 outgoing edges with DIFFERENT condition labels.
2. Every branch from a decision must eventually reconnect to a shared downstream step or an "end" node.
3. Sequential action steps must be connected by edges in order. No gaps.
4. No fabricated edges — every edge must correspond to a transition described in the source text.
5. No duplicate edges (same from/to/condition).
6. No dangling branches — every path from "start" must reach an "end" node.

If you find issues, return ONLY the corrected JSON with the fixes applied.
If the graph is correct, return it unchanged.
Return ONLY valid JSON. No markdown, no explanation.
"""


def review_edges(
    extracted: dict,
    source_text: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Run an edge-focused review pass on an extracted protocol graph.

    Uses a cheaper model (GPT-4o-mini) to specifically check edge
    completeness and correctness, fixing issues like:
    - Decision nodes with <2 outgoing edges
    - Missing sequential connections
    - Dangling branches
    """
    client = OpenAI()

    # Strip internal metadata before sending to reviewer
    graph_for_review = {k: v for k, v in extracted.items() if not k.startswith("_")}

    messages = [
        {"role": "system", "content": EDGE_REVIEW_PROMPT},
        {
            "role": "user",
            "content": (
                f"Review and fix the edges in this protocol graph.\n\n"
                f"--- SOURCE TEXT ---\n{source_text}\n\n"
                f"--- EXTRACTED GRAPH ---\n{json.dumps(graph_for_review, indent=2)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        return extracted  # Edge review returned empty, keep original
    try:
        reviewed = _safe_json_loads(raw)
    except (json.JSONDecodeError, ValueError):
        return extracted  # Edge review returned unparseable JSON, keep original

    # Safeguard: only accept the reviewed version if it doesn't introduce
    # more structural issues than the original
    original_issues = _count_edge_issues(extracted)
    reviewed_issues = _count_edge_issues(reviewed)

    if reviewed_issues <= original_issues:
        # Preserve original metadata
        for key in extracted:
            if key.startswith("_"):
                reviewed[key] = extracted[key]
        reviewed.setdefault("_extraction_meta", {})["edge_review"] = {
            "model": model,
            "original_issues": original_issues,
            "reviewed_issues": reviewed_issues,
            "accepted": True,
            "review_tokens": response.usage.total_tokens,
        }
        return reviewed
    else:
        extracted.setdefault("_extraction_meta", {})["edge_review"] = {
            "model": model,
            "original_issues": original_issues,
            "reviewed_issues": reviewed_issues,
            "accepted": False,
            "review_tokens": response.usage.total_tokens,
        }
        return extracted


def _count_edge_issues(graph: dict) -> int:
    """Count basic edge structural issues in a graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {n["id"] for n in nodes}
    issues = 0

    # Dangling edges
    for e in edges:
        if e.get("from") not in node_ids:
            issues += 1
        if e.get("to") not in node_ids:
            issues += 1

    # Decision nodes with <2 outgoing
    for n in nodes:
        if n.get("type") == "decision":
            outgoing = sum(1 for e in edges if e.get("from") == n["id"])
            if outgoing < 2:
                issues += 1

    # Orphan nodes
    connected = set()
    for e in edges:
        connected.add(e.get("from"))
        connected.add(e.get("to"))
    issues += len(node_ids - connected)

    return issues


def save_extraction(result: dict, output_path: str | Path) -> None:
    """Save an extraction result to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
