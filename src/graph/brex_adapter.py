"""
BREX Dataset Adapter

Converts the BREX (Business Rule EXtraction Benchmark) dataset into the
protocol graph format used by our pipeline, enabling evaluation against
published baselines across 13 LLMs.

BREX source: Yang et al. 2025, "Business as Rulesual"
  - 409 documents, 2,855 expert-annotated rules, 30+ domains
  - Rules: <<condition, operator, values>, action> pairs
  - Dependencies: Sequential, Conditional, Parallel

Mapping to our schema:
  Rule action text → action node
  Condition with "等于" (equals) → decision node (branching)
  Sequential dependency → edge
  Conditional dependency → edge with condition label
  Parallel dependency → edge (parallel split)

Note: BREX source texts are in Chinese. GPT-4o handles multilingual input,
so we pass the text directly to extraction.

Usage:
    from src.graph.brex_adapter import BREXAdapter

    adapter = BREXAdapter()
    adapter.download()
    protocols = adapter.convert_all()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


class BREXAdapter:
    """Adapter to convert BREX dataset to our protocol graph schema."""

    def __init__(self, cache_dir: str = ".cache/brex"):
        self._cache_dir = Path(cache_dir)
        self._dataset: list[dict] = []

    def download(self):
        """Download BREX dataset from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Install huggingface_hub: pip install huggingface_hub"
            )

        path = hf_hub_download(
            "XiaopiYu/BREX",
            "BREX.jsonl",
            repo_type="dataset",
        )
        self._dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._dataset.append(json.loads(line))

        print(f"Loaded BREX dataset: {len(self._dataset)} documents")

    def convert_all(self) -> list[dict]:
        """Convert all BREX documents to protocol graph format."""
        if not self._dataset:
            self.download()

        protocols = []
        for doc in self._dataset:
            protocol = self._convert_document(doc)
            if protocol and len(protocol.get("nodes", [])) >= 2:
                protocols.append(protocol)

        print(f"Converted {len(protocols)} documents to protocol graphs")
        return protocols

    @staticmethod
    def _parse_rules(rules_text: str) -> list[dict]:
        """Parse BREX rule pairs from text.

        Format: <<condition_field, operator, values>, action_text>
        Each rule on a separate line.
        """
        rules = []
        for line in rules_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match <<condition_field, operator, values>, action>
            m = re.match(
                r"<<([^,]+),\s*([^,]+),\s*([^>]+)>\s*,\s*(.+?)>\s*$", line
            )
            if m:
                rules.append({
                    "condition_field": m.group(1).strip(),
                    "operator": m.group(2).strip(),
                    "values": m.group(3).strip(),
                    "action": m.group(4).strip(),
                })
            else:
                # Try simpler match: <<field, operator, values>, action>
                m2 = re.match(r"<<(.+?)>,\s*(.+?)>?\s*$", line)
                if m2:
                    cond_part = m2.group(1).strip()
                    action_part = m2.group(2).strip().rstrip(">")
                    parts = cond_part.split(",", 2)
                    if len(parts) == 3:
                        rules.append({
                            "condition_field": parts[0].strip(),
                            "operator": parts[1].strip(),
                            "values": parts[2].strip(),
                            "action": action_part,
                        })
        return rules

    @staticmethod
    def _parse_dependencies(deps_text: str) -> list[dict]:
        """Parse BREX dependencies from text.

        Format: <<rule1> -> <<rule2>：relationship_type
        Relationship types: 顺序关系 (Sequential), 选择关系 (Conditional),
                           并行关系 (Parallel)
        """
        deps = []
        dep_type_map = {
            "顺序关系": "sequential",
            "选择关系": "conditional",
            "并行关系": "parallel",
        }

        for line in deps_text.strip().split("\n"):
            line = line.strip()
            if not line or "->" not in line:
                continue

            # Split on -> to get source and target
            parts = line.split("->", 1)
            if len(parts) != 2:
                continue

            source_part = parts[0].strip()
            target_and_type = parts[1].strip()

            # Extract dependency type (after ：or :)
            dep_type = "sequential"  # default
            for cn_type, en_type in dep_type_map.items():
                if cn_type in target_and_type:
                    dep_type = en_type
                    target_and_type = target_and_type.split(
                        cn_type)[0].strip().rstrip("：").rstrip(":")
                    break

            # Extract rule references from source and target
            source_rule = BREXAdapter._extract_rule_ref(source_part)
            target_rule = BREXAdapter._extract_rule_ref(target_and_type)

            if source_rule and target_rule:
                deps.append({
                    "source": source_rule,
                    "target": target_rule,
                    "type": dep_type,
                })
        return deps

    @staticmethod
    def _extract_rule_ref(text: str) -> Optional[str]:
        """Extract a rule reference string for matching.

        Converts <<condition, operator, values>, action> to a canonical key.
        """
        text = text.strip()
        # Match <<X>, Y> pattern
        m = re.match(r"<<(.+?)>,\s*(.+?)>?\s*$", text)
        if m:
            return f"<<{m.group(1).strip()}>, {m.group(2).strip().rstrip('>')}>"
        return None

    def _convert_document(self, doc: dict) -> Optional[dict]:
        """Convert a single BREX document to protocol graph format."""
        domain = doc.get("领域", "unknown")
        intent = doc.get("意图", "unknown")
        text = doc.get("文本", "")
        rules_text = doc.get("业务规则二元组", "")
        deps_text = doc.get("依赖关系", "")

        if not text or not rules_text:
            return None

        doc_name = f"{domain}_{intent}"

        rules = self._parse_rules(rules_text)
        deps = self._parse_dependencies(deps_text)

        if not rules:
            return None

        # Build nodes from rules
        nodes = []
        node_id_counter = 1
        rule_to_node_id: dict[str, str] = {}

        # Add start node
        nodes.append({
            "id": f"n{node_id_counter}",
            "type": "start",
            "text": f"{intent} process started",
        })
        node_id_counter += 1

        for rule in rules:
            rule_key = f"<<{rule['condition_field']}, {rule['operator']}, {rule['values']}>, {rule['action']}>"
            node_id = f"n{node_id_counter}"

            # Determine node type based on operator
            if rule["operator"] == "等于":
                # Equality check → this represents a decision branch
                node_type = "decision"
                node_text = (
                    f"{rule['condition_field']} {rule['operator']} "
                    f"{rule['values']} → {rule['action']}"
                )
            else:
                node_type = "action"
                node_text = (
                    f"{rule['action']} "
                    f"({rule['condition_field']} {rule['operator']} {rule['values']})"
                )

            # Handle "None" action (terminal rule)
            if rule["action"] == "None" or rule["action"] == "none":
                node_type = "end"
                node_text = f"Process complete ({rule['condition_field']})"

            nodes.append({
                "id": node_id,
                "type": node_type,
                "text": node_text,
            })
            rule_to_node_id[rule_key] = node_id
            node_id_counter += 1

        # If no explicit end node, add one
        has_end = any(n["type"] == "end" for n in nodes)
        if not has_end:
            nodes.append({
                "id": f"n{node_id_counter}",
                "type": "end",
                "text": "Process complete",
            })
            node_id_counter += 1

        # Build edges from dependencies
        edges = []
        edge_id = 1

        for dep in deps:
            source_id = rule_to_node_id.get(dep["source"])
            target_id = rule_to_node_id.get(dep["target"])

            if source_id and target_id:
                edge = {
                    "id": f"e{edge_id}",
                    "from": source_id,
                    "to": target_id,
                }
                if dep["type"] == "conditional":
                    edge["condition"] = "conditional branch"
                elif dep["type"] == "parallel":
                    edge["condition"] = "parallel"

                edges.append(edge)
                edge_id += 1

        # Connect start to first rule nodes (those that appear only as sources)
        target_ids = {dep["target"] for dep in deps}
        source_ids = {dep["source"] for dep in deps}
        root_rules = source_ids - target_ids
        for root_key in root_rules:
            root_node_id = rule_to_node_id.get(root_key)
            if root_node_id:
                edges.append({
                    "id": f"e{edge_id}",
                    "from": "n1",  # start node
                    "to": root_node_id,
                })
                edge_id += 1

        return {
            "title": doc_name,
            "source_pdf": "BREX_dataset",
            "nodes": nodes,
            "edges": edges,
            "_brex_meta": {
                "domain": domain,
                "intent": intent,
                "num_rules": len(rules),
                "num_deps": len(deps),
            },
        }

    def save_as_gold(self, output_dir: str, max_docs: int = 0):
        """Save converted protocols as gold standard JSON files."""
        protocols = self.convert_all()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if max_docs > 0:
            protocols = protocols[:max_docs]

        for protocol in protocols:
            name = protocol["title"].replace(" ", "_").replace("/", "_")
            fname = out_path / f"{name}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(protocol, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(protocols)} gold files to {output_dir}")
