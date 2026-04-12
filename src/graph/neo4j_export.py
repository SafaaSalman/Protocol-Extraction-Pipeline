"""
Neo4j Graph Exporter

Exports extracted protocol graphs to a Neo4j database.
Supports both individual protocol loading and batch loading.

Node labels:
  - Protocol         (top-level protocol metadata)
  - Step             (action/start/end nodes within a protocol)
  - Decision         (decision nodes within a protocol)

Relationship types:
  - HAS_STEP         (Protocol -> Step/Decision)
  - NEXT             (Step -> Step, sequential flow)
  - BRANCHES_TO      (Decision -> Step, with condition property)
  - CROSS_REFERENCES (Protocol -> Protocol, when one references another)

Usage:
    from src.graph.neo4j_export import Neo4jExporter

    exporter = Neo4jExporter("bolt://localhost:7687", "neo4j", "password")
    exporter.load_protocol("evaluation/extracted/cpr.json")
    exporter.load_all("evaluation/extracted/")
    exporter.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase


class Neo4jExporter:
    """Export protocol graphs to Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Schema setup ─────────────────────────────────────────────────

    def create_constraints(self):
        """Create uniqueness constraints and indexes for efficient querying."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protocol) REQUIRE p.protocol_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Step) REQUIRE s.uid IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (s:Step) ON (s.node_type)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Step) ON (s.protocol_id)",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in constraints:
                session.run(stmt)

    # ── Protocol loading ─────────────────────────────────────────────

    def load_protocol(self, json_path: str | Path) -> dict:
        """Load a single protocol JSON file into Neo4j.

        Returns summary dict with counts of nodes/edges created.
        """
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._import_protocol(data)

    def load_all(self, directory: str | Path, pattern: str = "*.json") -> list[dict]:
        """Load all protocol JSON files from a directory."""
        directory = Path(directory)
        results = []
        for json_file in sorted(directory.glob(pattern)):
            # Skip manifest and non-protocol files
            if json_file.name in ("manifest.json",):
                continue
            try:
                result = self.load_protocol(json_file)
                results.append(result)
                print(f"  Loaded {json_file.name}: {result['nodes_created']} nodes, "
                      f"{result['edges_created']} edges")
            except Exception as e:
                print(f"  ERROR loading {json_file.name}: {e}")
                results.append({"file": json_file.name, "error": str(e)})
        return results

    def _import_protocol(self, data: dict) -> dict:
        """Import a single protocol graph dict into Neo4j."""
        protocol_id = data.get("protocol_id", "unknown")

        with self._driver.session(database=self._database) as session:
            # 1. Create Protocol node
            session.run(
                """
                MERGE (p:Protocol {protocol_id: $pid})
                SET p.title = $title,
                    p.source_pdf = $source_pdf,
                    p.pages = $pages,
                    p.protocol_type = $protocol_type,
                    p.extraction_confidence = $confidence
                """,
                pid=protocol_id,
                title=data.get("title", ""),
                source_pdf=data.get("source_pdf", ""),
                pages=data.get("pages", []),
                protocol_type=data.get("protocol_type", "unknown"),
                confidence=data.get("extraction_confidence", 0.0),
            )

            # 2. Create Step/Decision nodes
            nodes_created = 0
            for node in data.get("nodes", []):
                uid = f"{protocol_id}__{node['id']}"
                node_type = node.get("type", "action")
                label = "Decision" if node_type == "decision" else "Step"

                session.run(
                    f"""
                    MERGE (s:{label} {{uid: $uid}})
                    SET s.node_id = $nid,
                        s.protocol_id = $pid,
                        s.node_type = $ntype,
                        s.text = $text,
                        s.evidence_page = $epage,
                        s.evidence_source_text = $esrc
                    WITH s
                    MATCH (p:Protocol {{protocol_id: $pid}})
                    MERGE (p)-[:HAS_STEP]->(s)
                    """,
                    uid=uid,
                    nid=node["id"],
                    pid=protocol_id,
                    ntype=node_type,
                    text=node.get("text", ""),
                    epage=node.get("evidence", {}).get("page"),
                    esrc=node.get("evidence", {}).get("source_text", ""),
                )
                nodes_created += 1

            # 3. Create edges (NEXT / BRANCHES_TO)
            edges_created = 0
            for edge in data.get("edges", []):
                from_uid = f"{protocol_id}__{edge['from']}"
                to_uid = f"{protocol_id}__{edge['to']}"
                condition = edge.get("condition", "")

                # Find the source node to determine relationship type
                from_node = next(
                    (n for n in data.get("nodes", []) if n["id"] == edge["from"]),
                    None,
                )
                rel_type = "BRANCHES_TO" if (from_node and from_node.get("type") == "decision") else "NEXT"

                session.run(
                    f"""
                    MATCH (a {{uid: $from_uid}})
                    MATCH (b {{uid: $to_uid}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.condition = $condition
                    """,
                    from_uid=from_uid,
                    to_uid=to_uid,
                    condition=condition,
                )
                edges_created += 1

            # 4. Cross-reference edges between protocols
            cross_refs = data.get("cross_references", [])
            for ref in cross_refs:
                session.run(
                    """
                    MATCH (p:Protocol {protocol_id: $pid})
                    MERGE (ref:Protocol {title: $ref_title})
                    MERGE (p)-[:CROSS_REFERENCES]->(ref)
                    """,
                    pid=protocol_id,
                    ref_title=ref,
                )

            # 5. Load enrichment entities if present
            entities_created = 0
            enrichment = data.get("enrichment", {})
            if enrichment:
                entities_created = self._load_enrichment(
                    session, protocol_id, enrichment, data
                )

        return {
            "file": data.get("title", protocol_id),
            "protocol_id": protocol_id,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "cross_references": len(cross_refs),
            "entities_created": entities_created,
        }

    def _load_enrichment(
        self, session, protocol_id: str, enrichment: dict, data: dict
    ) -> int:
        """Load semantic entities from enrichment into Neo4j.

        Creates entity nodes (Role, Equipment, Hazard, Condition, IncidentType)
        and links them to the Protocol and individual Step nodes.
        """
        entities_created = 0
        global_entities = enrichment.get("entities", {})

        # Map entity category to Neo4j label and relationship type
        entity_config = {
            "roles": ("Role", "PERFORMED_BY"),
            "equipment": ("Equipment", "REQUIRES"),
            "hazards": ("Hazard", "INVOLVES_HAZARD"),
            "conditions": ("Condition", "HAS_CONDITION"),
            "incident_types": ("IncidentType", "APPLIES_TO"),
        }

        # Create global entity nodes linked to Protocol
        for category, (label, rel_type) in entity_config.items():
            for entity_name in global_entities.get(category, []):
                session.run(
                    f"""
                    MERGE (e:{label} {{name: $name}})
                    WITH e
                    MATCH (p:Protocol {{protocol_id: $pid}})
                    MERGE (p)-[:{rel_type}]->(e)
                    """,
                    name=entity_name.lower().strip(),
                    pid=protocol_id,
                )
                entities_created += 1

        # Link entities to individual step nodes
        node_annotations = enrichment.get("node_annotations", {})
        for node_id, annotations in node_annotations.items():
            uid = f"{protocol_id}__{node_id}"
            for category, (label, rel_type) in entity_config.items():
                for entity_name in annotations.get(category, []):
                    session.run(
                        f"""
                        MATCH (s {{uid: $uid}})
                        MERGE (e:{label} {{name: $name}})
                        MERGE (s)-[:{rel_type}]->(e)
                        """,
                        uid=uid,
                        name=entity_name.lower().strip(),
                    )

        return entities_created

    # ── Queries ──────────────────────────────────────────────────────

    def clear_database(self):
        """Delete all nodes and relationships. Use with caution."""
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_protocol(self, protocol_id: str) -> Optional[dict]:
        """Retrieve a protocol and its graph from Neo4j."""
        with self._driver.session(database=self._database) as session:
            # Get protocol metadata
            result = session.run(
                "MATCH (p:Protocol {protocol_id: $pid}) RETURN p",
                pid=protocol_id,
            )
            record = result.single()
            if not record:
                return None

            proto = dict(record["p"])

            # Get nodes
            nodes_result = session.run(
                """
                MATCH (p:Protocol {protocol_id: $pid})-[:HAS_STEP]->(s)
                RETURN s ORDER BY s.node_id
                """,
                pid=protocol_id,
            )
            proto["nodes"] = [dict(r["s"]) for r in nodes_result]

            # Get edges
            edges_result = session.run(
                """
                MATCH (p:Protocol {protocol_id: $pid})-[:HAS_STEP]->(a)
                MATCH (a)-[r:NEXT|BRANCHES_TO]->(b)
                RETURN a.node_id AS from_id, b.node_id AS to_id,
                       r.condition AS condition, type(r) AS rel_type
                """,
                pid=protocol_id,
            )
            proto["edges"] = [
                {
                    "from": r["from_id"],
                    "to": r["to_id"],
                    "condition": r["condition"] or "",
                    "type": r["rel_type"],
                }
                for r in edges_result
            ]

            return proto

    def list_protocols(self) -> list[dict]:
        """List all protocols in the database."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (p:Protocol)
                OPTIONAL MATCH (p)-[:HAS_STEP]->(s)
                RETURN p.protocol_id AS id, p.title AS title,
                       p.protocol_type AS type, count(s) AS node_count
                ORDER BY p.title
                """
            )
            return [dict(r) for r in result]

    def find_protocols_by_keyword(self, keyword: str) -> list[dict]:
        """Find protocols containing a keyword in any step text."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (p:Protocol)-[:HAS_STEP]->(s)
                WHERE toLower(s.text) CONTAINS toLower($keyword)
                RETURN DISTINCT p.protocol_id AS id, p.title AS title,
                       collect(s.text)[..3] AS matching_steps
                """,
                keyword=keyword,
            )
            return [dict(r) for r in result]

    def find_cross_protocol_paths(self, from_protocol: str, to_protocol: str) -> list:
        """Find paths between two protocols via cross-references."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH path = (a:Protocol {protocol_id: $from_id})
                    -[:CROSS_REFERENCES*1..3]-
                    (b:Protocol {protocol_id: $to_id})
                RETURN [n IN nodes(path) | n.title] AS protocol_chain
                LIMIT 5
                """,
                from_id=from_protocol,
                to_id=to_protocol,
            )
            return [r["protocol_chain"] for r in result]

    def get_graph_stats(self) -> dict:
        """Get summary statistics of the knowledge graph."""
        with self._driver.session(database=self._database) as session:
            stats = {}
            result = session.run("MATCH (p:Protocol) RETURN count(p) AS c")
            stats["protocols"] = result.single()["c"]

            result = session.run("MATCH (s:Step) RETURN count(s) AS c")
            stats["steps"] = result.single()["c"]

            result = session.run("MATCH (d:Decision) RETURN count(d) AS c")
            stats["decisions"] = result.single()["c"]

            result = session.run("MATCH ()-[r:NEXT]->() RETURN count(r) AS c")
            stats["next_edges"] = result.single()["c"]

            result = session.run("MATCH ()-[r:BRANCHES_TO]->() RETURN count(r) AS c")
            stats["branch_edges"] = result.single()["c"]

            result = session.run("MATCH ()-[r:CROSS_REFERENCES]->() RETURN count(r) AS c")
            stats["cross_references"] = result.single()["c"]

            return stats
