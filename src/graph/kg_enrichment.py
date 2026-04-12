"""
Knowledge Graph Enrichment

Second LLM pass over extracted protocols to identify semantic entities
(roles, equipment, hazards, conditions, incident types) and relations.
Enriches the protocol graph with domain-specific annotations.

Usage:
    from src.graph.kg_enrichment import enrich_protocol

    enriched = enrich_protocol(protocol_graph, model="gpt-4o-mini")
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI


ENRICHMENT_PROMPT = """\
You are a domain expert in emergency management protocols. Given a protocol \
graph (nodes and edges), extract semantic entities and relations.

For each node, identify:
- roles: people/positions involved (e.g., "Paramedic", "Incident Commander")
- equipment: tools or resources mentioned (e.g., "AED", "PPE", "radio")
- hazards: dangers or risks referenced (e.g., "fire", "smoke inhalation")
- conditions: preconditions or environmental factors (e.g., "patient unconscious")
- incident_types: types of incidents this applies to (e.g., "cardiac arrest", "wildfire")

Return a JSON object with:
{
  "entities": {
    "roles": ["role1", "role2"],
    "equipment": ["item1", "item2"],
    "hazards": ["hazard1"],
    "conditions": ["condition1"],
    "incident_types": ["type1"]
  },
  "node_annotations": {
    "n1": {"roles": [], "equipment": [], "hazards": []},
    "n2": {"roles": ["Paramedic"], "equipment": ["AED"]}
  }
}

Rules:
- Only extract entities explicitly mentioned in node text or edge conditions
- Normalize names (lowercase, singular form)
- Do not invent entities not present in the protocol
- Include node_annotations only for nodes that have entities
- Return ONLY valid JSON"""


def enrich_protocol(
    protocol: dict,
    model: str = "gpt-4o-mini",
) -> dict:
    """Enrich a protocol graph with semantic entity annotations.

    Adds an 'enrichment' key to the protocol dict containing extracted
    entities and per-node annotations.
    """
    client = OpenAI()

    # Build a compact representation for the LLM
    compact = {
        "title": protocol.get("title", ""),
        "protocol_type": protocol.get("protocol_type", ""),
        "nodes": [
            {"id": n["id"], "type": n["type"], "text": n["text"]}
            for n in protocol.get("nodes", [])
        ],
        "edges": [
            {"from": e["from"], "to": e["to"], "condition": e.get("condition", "")}
            for e in protocol.get("edges", [])
        ],
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ENRICHMENT_PROMPT},
            {"role": "user", "content": json.dumps(compact, indent=2)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    enrichment = json.loads(response.choices[0].message.content)
    tokens = response.usage.total_tokens if response.usage else 0

    protocol["enrichment"] = enrichment
    protocol.setdefault("_extraction_meta", {})["enrichment_tokens"] = tokens

    return protocol


def enrich_batch(
    directory: str,
    output_dir: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Enrich all protocol JSON files in a directory.

    If output_dir is None, updates files in-place.
    """
    from pathlib import Path

    directory = Path(directory)
    out_dir = Path(output_dir) if output_dir else directory
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for json_file in sorted(directory.glob("*.json")):
        if json_file.name in ("manifest.json",):
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            protocol = json.load(f)

        # Skip if already enriched
        if "enrichment" in protocol:
            print(f"  Skipping {json_file.name} (already enriched)")
            continue

        try:
            enriched = enrich_protocol(protocol, model=model)
            out_path = out_dir / json_file.name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2)

            entities = enriched.get("enrichment", {}).get("entities", {})
            total = sum(len(v) for v in entities.values())
            print(f"  Enriched {json_file.name}: {total} entities found")
            results.append({"file": json_file.name, "entities": total})
        except Exception as e:
            print(f"  ERROR enriching {json_file.name}: {e}")
            results.append({"file": json_file.name, "error": str(e)})

    return results
