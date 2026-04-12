"""
Protocol Extraction Module

Uses OpenAI GPT-4o to extract structured protocol graphs from PDF section text.
Supports few-shot prompting with gold standard examples.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.ingestion.pdf_ingest import DocumentContent, render_page_image
from src.extraction.section_detector import ProtocolSection


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

1. Every protocol must have exactly one "start" node and at least one "end" node.
2. Every "decision" node must have at least 2 outgoing edges with different conditions.
3. "action" nodes represent imperative steps (things to do).
4. "decision" nodes represent questions or conditional checks (yes/no, if/then).
5. Edge conditions should be clear and descriptive (e.g., "Yes (breathing)", "No (not breathing)").
6. If the protocol has numbered steps, preserve the sequential order.
7. If there are branching paths (if/then, yes/no), model them as decision nodes with conditional edges.
8. If the protocol references other protocols or pages, include them in cross_references.
9. Every node must have evidence linking back to the source text and page number.
10. Set extraction_confidence between 0.0 and 1.0 based on how clear and unambiguous the protocol is.
11. Use protocol_type "sequential" for linear step lists, "decision_tree" for branching logic, "table" for table-based protocols, "hybrid" for mixed formats.
12. Node IDs must be sequential: n1, n2, n3, etc.
13. No orphan nodes — every node must be connected via at least one edge (except isolated start with one outgoing or end with one incoming).
14. Do NOT invent steps that are not in the source text. Only extract what is explicitly stated.
15. Return ONLY valid JSON. No markdown, no explanation, no code fences.
"""


def _load_few_shot_examples() -> list[dict]:
    """Load gold standard files to use as few-shot examples."""
    gold_dir = Path(__file__).parent.parent.parent / "evaluation" / "gold"
    # Use CPR (decision_tree) and Burn Injuries (sequential) as examples
    # to demonstrate both formats
    example_files = ["cpr_01.json", "burn_injuries_01.json"]
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

    # Build messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
    result = json.loads(raw)

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

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

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
    result = json.loads(raw)

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
    """
    client = OpenAI()

    section_text = _gather_section_text(section, doc)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if use_few_shot:
        examples = _load_few_shot_examples()
        messages.extend(_build_few_shot_messages(examples))

    # Build multimodal user message with text + images
    context_line = f"\n{toc_context}\n" if toc_context else ""

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
                f"Pay special attention to arrows, branching paths, Yes/No labels, and color coding.\n\n"
                f"--- SECTION TEXT ---\n{section_text}"
            ),
        }
    ]

    # Render and attach each page image
    for pg_num in range(section.start_page, section.end_page + 1):
        img_bytes = render_page_image(pdf_path, pg_num, dpi=dpi)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
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
    result = json.loads(raw)

    result["_extraction_meta"] = {
        "model": model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "few_shot": use_few_shot,
        "multimodal": True,
        "dpi": dpi,
    }

    return result


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


def save_extraction(result: dict, output_path: str | Path) -> None:
    """Save an extraction result to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
