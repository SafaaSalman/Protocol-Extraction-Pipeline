"""
PET Dataset Adapter

Converts the PET (Process Extraction from Text) dataset into the protocol
graph format used by our pipeline, enabling cross-domain evaluation.

PET entities → our node types:
  Activity → action
  XOR Gateway → decision
  AND Gateway → action (parallel split)
  Actor → role annotation on connected action nodes

PET relations → our edge types:
  flow → edge (condition from Condition Specification if present)
  uses → enrichment relation
  actor performer/recipient → enrichment relation

Usage:
    from src.graph.pet_adapter import PETAdapter

    adapter = PETAdapter()
    adapter.download()
    protocols = adapter.convert_all()
    adapter.save_as_gold("evaluation/pet_gold/")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


class PETAdapter:
    """Adapter to convert PET dataset to our protocol graph schema."""

    def __init__(self, cache_dir: str = ".cache/pet"):
        self._cache_dir = Path(cache_dir)
        self._dataset = None

    def download(self):
        """Download PET dataset from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Install huggingface_hub and pandas: pip install huggingface_hub pandas"
            )

        path = hf_hub_download(
            "patriziobellan/PET",
            "data/test-00000-of-00001-4cd746ae057084a3.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        # Convert DataFrame rows to list of dicts matching expected format
        self._dataset = [row.to_dict() for _, row in df.iterrows()]
        print(f"Loaded PET dataset: {len(self._dataset)} documents")

    def convert_all(self) -> list[dict]:
        """Convert all PET documents to protocol graph format."""
        if self._dataset is None:
            self.download()

        protocols = []
        for doc in self._dataset:
            protocol = self._convert_document(doc)
            if protocol and len(protocol.get("nodes", [])) >= 2:
                protocols.append(protocol)

        print(f"Converted {len(protocols)} documents to protocol graphs")
        return protocols

    def _convert_document(self, doc: dict) -> Optional[dict]:
        """Convert a single PET document to protocol graph format."""
        doc_name = doc["document name"]
        tokens = doc["tokens"]
        ner_tags = doc["ner_tags"]
        sentence_ids = doc["sentence-IDs"]
        token_ids = doc["tokens-IDs"]

        # 1. Extract entities from BIO tags
        entities = self._extract_entities(tokens, ner_tags, sentence_ids, token_ids)

        if not entities:
            return None

        # 2. Extract relations
        relations = self._extract_relations(doc, entities)

        # 3. Build protocol graph
        nodes = []
        node_map = {}  # entity_key -> node_id
        node_counter = 1

        # Add start node
        nodes.append({
            "id": "n1",
            "type": "start",
            "text": f"{doc_name} process started",
        })
        node_counter += 1

        # Convert entities to nodes
        for entity in entities:
            etype = entity["type"]
            if etype in ("Activity", "XOR Gateway", "AND Gateway"):
                nid = f"n{node_counter}"
                node_type = "decision" if etype == "XOR Gateway" else "action"
                nodes.append({
                    "id": nid,
                    "type": node_type,
                    "text": entity["text"],
                    "evidence": {
                        "page": 1,
                        "source_text": entity["text"],
                    },
                })
                node_map[entity["key"]] = nid
                node_counter += 1

        # Add end node
        end_id = f"n{node_counter}"
        nodes.append({
            "id": end_id,
            "type": "end",
            "text": "Process complete",
        })

        # 4. Build edges from flow relations
        edges = []
        connected_nodes = set()

        for rel in relations:
            if rel["type"] == "flow":
                src_key = rel["source_key"]
                tgt_key = rel["target_key"]
                if src_key in node_map and tgt_key in node_map:
                    condition = rel.get("condition", "")
                    edges.append({
                        "from": node_map[src_key],
                        "to": node_map[tgt_key],
                        "condition": condition,
                    })
                    connected_nodes.add(node_map[src_key])
                    connected_nodes.add(node_map[tgt_key])

        # Connect start to first activity if not connected
        activity_nodes = [n for n in nodes if n["type"] in ("action", "decision")]
        if activity_nodes:
            edges.insert(0, {"from": "n1", "to": activity_nodes[0]["id"]})

        # Connect last unconnected node to end
        outgoing = {e["from"] for e in edges}
        for node in reversed(activity_nodes):
            if node["id"] not in outgoing:
                edges.append({"from": node["id"], "to": end_id})
                break

        # 5. Determine protocol type
        has_decisions = any(n["type"] == "decision" for n in nodes)
        protocol_type = "decision_tree" if has_decisions else "sequential"

        # 6. Build enrichment from actors and activity data
        enrichment = self._build_enrichment(entities, relations, node_map)

        protocol = {
            "protocol_id": doc_name.replace(" ", "_").lower(),
            "title": doc_name,
            "source_pdf": "PET_dataset",
            "pages": [1],
            "protocol_type": protocol_type,
            "extraction_confidence": 1.0,
            "cross_references": [],
            "nodes": nodes,
            "edges": edges,
        }

        if enrichment:
            protocol["enrichment"] = enrichment

        return protocol

    def _extract_entities(
        self,
        tokens: list[str],
        ner_tags: list[str],
        sentence_ids: list[int],
        token_ids: list[int],
    ) -> list[dict]:
        """Extract named entities from BIO-tagged tokens."""
        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    "type": entity_type,
                    "tokens": [token],
                    "start_sentence": sentence_ids[i],
                    "start_token": token_ids[i],
                    "key": f"{sentence_ids[i]}_{token_ids[i]}",
                }
            elif tag.startswith("I-") and current_entity:
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    current_entity["text"] = " ".join(current_entity["tokens"])
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            current_entity["text"] = " ".join(current_entity["tokens"])
            entities.append(current_entity)

        # Add text to all entities
        for entity in entities:
            if "text" not in entity:
                entity["text"] = " ".join(entity.get("tokens", []))

        return entities

    def _extract_relations(self, doc: dict, entities: list[dict]) -> list[dict]:
        """Extract relations from the document."""
        relations_data = doc.get("relations", {})
        if not relations_data:
            return []

        source_sids = relations_data.get("source-head-sentence-ID", [])
        source_wids = relations_data.get("source-head-word-ID", [])
        rel_types = relations_data.get("relation-type", [])
        target_sids = relations_data.get("target-head-sentence-ID", [])
        target_wids = relations_data.get("target-head-word-ID", [])

        relations = []
        for i in range(len(rel_types)):
            src_key = f"{source_sids[i]}_{source_wids[i]}"
            tgt_key = f"{target_sids[i]}_{target_wids[i]}"

            relations.append({
                "type": rel_types[i],
                "source_key": src_key,
                "target_key": tgt_key,
            })

        return relations

    def _build_enrichment(
        self,
        entities: list[dict],
        relations: list[dict],
        node_map: dict,
    ) -> dict:
        """Build enrichment dict from actors and activity data."""
        roles = set()
        equipment = set()

        for entity in entities:
            if entity["type"] == "Actor":
                roles.add(entity["text"].lower())
            elif entity["type"] == "Activity Data":
                equipment.add(entity["text"].lower())

        if not roles and not equipment:
            return {}

        return {
            "entities": {
                "roles": sorted(roles),
                "equipment": sorted(equipment),
                "hazards": [],
                "conditions": [],
                "incident_types": [],
            },
            "node_annotations": {},
        }

    def save_as_gold(self, output_dir: str, protocols: Optional[list[dict]] = None):
        """Save converted protocols as gold standard JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if protocols is None:
            protocols = self.convert_all()

        for protocol in protocols:
            safe_name = protocol["protocol_id"]
            safe_name = re.sub(r"[^a-z0-9_]", "_", safe_name)
            out_path = output_dir / f"{safe_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(protocol, f, indent=2)

        print(f"Saved {len(protocols)} PET protocols to {output_dir}/")


def evaluate_on_pet(extractor_fn, num_samples: int = 10) -> dict:
    """Evaluate our extraction pipeline against PET gold standards.

    Args:
        extractor_fn: Callable that takes (title, text, pages, source_pdf) and
                      returns a protocol graph dict.
        num_samples: Number of PET documents to evaluate on.

    Returns:
        Evaluation summary dict.
    """
    import sys
    sys.path.insert(0, ".")
    from evaluate import full_evaluation

    adapter = PETAdapter()
    protocols = adapter.convert_all()[:num_samples]

    results = []
    for gold in protocols:
        # Reconstruct the source text from node texts
        source_text = " ".join(
            n["text"] for n in gold["nodes"]
            if n["type"] not in ("start", "end")
        )

        try:
            extracted = extractor_fn(
                title=gold["title"],
                section_text=source_text,
                pages=[1],
                source_pdf="PET_dataset",
            )

            eval_result = full_evaluation(gold, extracted)
            results.append({
                "document": gold["protocol_id"],
                "evaluation": eval_result,
            })
        except Exception as e:
            results.append({
                "document": gold["protocol_id"],
                "error": str(e),
            })

    # Compute averages
    valid = [r for r in results if "evaluation" in r]
    if valid:
        avg_node_f1 = sum(
            r["evaluation"]["nodes"]["f1"] for r in valid
        ) / len(valid)
        avg_edge_f1 = sum(
            r["evaluation"]["edges"]["f1"] for r in valid
        ) / len(valid)
    else:
        avg_node_f1 = avg_edge_f1 = 0.0

    return {
        "num_evaluated": len(valid),
        "num_errors": len(results) - len(valid),
        "avg_node_f1": round(avg_node_f1, 3),
        "avg_edge_f1": round(avg_edge_f1, 3),
        "per_document": results,
    }
