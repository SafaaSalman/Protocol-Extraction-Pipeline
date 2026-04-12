"""
PyReason Temporal Reasoning for Protocol Graphs

Converts extracted protocol graphs into PyReason-compatible graphs
and applies temporal logic rules to:
- Verify protocol execution paths are complete
- Reason about step ordering and dependencies
- Check temporal constraints (e.g., "step A must precede step B")
- Answer reachability queries ("given condition X, which steps execute?")

Requires: pyreason >= 3.0, numba

Usage:
    from src.graph.pyreason_engine import ProtocolReasoner

    reasoner = ProtocolReasoner()
    reasoner.load_protocol("evaluation/extracted/cpr.json")
    results = reasoner.reason(num_timesteps=10)
    reasoner.query_reachability("n1")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx

# Optional pyreason import
try:
    import pyreason as pr
    HAS_PYREASON = True
except ImportError:
    HAS_PYREASON = False


class ProtocolReasoner:
    """Temporal logic reasoning over protocol graphs using PyReason."""

    def __init__(self):
        if not HAS_PYREASON:
            raise ImportError(
                "PyReason is not installed. Install with: pip install pyreason\n"
                "Note: PyReason requires Python 3.9-3.12."
            )
        self._graph = None
        self._protocol = None
        self._interpretation = None

    def load_protocol(self, json_path: str | Path) -> nx.DiGraph:
        """Load a protocol JSON and convert to a PyReason-compatible NetworkX graph."""
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            self._protocol = json.load(f)

        self._graph = self._protocol_to_networkx(self._protocol)
        return self._graph

    def load_protocol_dict(self, protocol: dict) -> nx.DiGraph:
        """Load a protocol from a dict."""
        self._protocol = protocol
        self._graph = self._protocol_to_networkx(protocol)
        return self._graph

    def _protocol_to_networkx(self, protocol: dict) -> nx.DiGraph:
        """Convert protocol graph JSON to a NetworkX DiGraph with PyReason attributes.

        Node attributes:
            - node_type: 1 for start, decision, etc.
            - is_start, is_end, is_decision, is_action: binary flags
            - active: initially 0 (will be set via facts)

        Edge attributes:
            - flow: 1 (sequential flow)
            - has_condition: 1 if edge has a non-empty condition
        """
        g = nx.DiGraph()

        for node in protocol.get("nodes", []):
            nid = node["id"]
            ntype = node.get("type", "action")
            g.add_node(
                nid,
                node_type=1,
                is_start=1 if ntype == "start" else 0,
                is_end=1 if ntype == "end" else 0,
                is_decision=1 if ntype == "decision" else 0,
                is_action=1 if ntype == "action" else 0,
                active=0,
            )

        for edge in protocol.get("edges", []):
            condition = edge.get("condition", "")
            g.add_edge(
                edge["from"],
                edge["to"],
                flow=1,
                has_condition=1 if condition else 0,
            )

        return g

    def setup_rules(self):
        """Define PyReason rules for protocol execution reasoning.

        Rules:
        1. Start activation: start nodes become active at t=0
        2. Forward propagation: if a node is active and has flow to next, next becomes active
        3. Decision propagation: decision nodes propagate to all branches
        4. Completion check: end node active means protocol is completable
        """
        pr.reset()
        pr.load_graph(self._graph)

        # Rule 1: Active nodes propagate forward along flow edges
        # If node x is active and there's a flow edge from x to y, then y becomes active
        pr.add_rule(
            pr.Rule(
                "active(y) <-1 active(x), flow(x,y)",
                "forward_propagation",
            )
        )

        # Rule 2: Decision nodes propagate to all outgoing branches
        pr.add_rule(
            pr.Rule(
                "active(y) <-1 is_decision(x), active(x), flow(x,y)",
                "decision_branch_propagation",
            )
        )

        # Set the start node as active via a fact
        start_nodes = [
            n["id"] for n in self._protocol.get("nodes", [])
            if n.get("type") == "start"
        ]
        for start_id in start_nodes:
            pr.add_fact(
                pr.Fact(
                    fact_text=f"active({start_id})",
                    name=f"start_activation_{start_id}",
                    start_time=0,
                    end_time=0,
                )
            )

        # Settings
        pr.settings.verbose = False

    def reason(self, num_timesteps: int = 10) -> dict:
        """Run PyReason inference and return results.

        Args:
            num_timesteps: Number of temporal steps to reason over.
                          Should be at least as large as the longest path.

        Returns:
            Dict with active nodes at each timestep and completeness info.
        """
        self.setup_rules()

        # Run reasoning
        self._interpretation = pr.reason(timesteps=num_timesteps)

        # Collect results
        results = self._collect_results()
        return results

    def _collect_results(self) -> dict:
        """Collect reasoning results into a structured dict."""
        if self._interpretation is None:
            return {}

        # Get all node interpretations
        node_df = self._interpretation.filter_and_sort_nodes(
            "active", tuple([1, 1])
        )

        active_nodes = set()
        activation_times = {}

        for _, row in node_df.iterrows():
            node_name = row["component"]
            t_lower = row["time"]
            active_nodes.add(node_name)
            if node_name not in activation_times:
                activation_times[node_name] = t_lower

        # Check completeness
        end_nodes = [
            n["id"] for n in self._protocol.get("nodes", [])
            if n.get("type") == "end"
        ]
        all_nodes = {n["id"] for n in self._protocol.get("nodes", [])}
        reachable = active_nodes & all_nodes
        unreachable = all_nodes - reachable
        ends_reached = [e for e in end_nodes if e in active_nodes]

        return {
            "protocol_id": self._protocol.get("protocol_id", ""),
            "total_nodes": len(all_nodes),
            "reachable_nodes": len(reachable),
            "unreachable_nodes": sorted(unreachable),
            "completion": len(ends_reached) > 0,
            "ends_reached": ends_reached,
            "activation_order": dict(
                sorted(activation_times.items(), key=lambda x: x[1])
            ),
        }

    def query_reachability(self, from_node: str) -> list[str]:
        """Query which nodes are reachable from a given node.

        Uses NetworkX directly (no PyReason needed for simple reachability).
        """
        if self._graph is None:
            return []

        reachable = nx.descendants(self._graph, from_node)
        return sorted(reachable)

    def find_critical_paths(self) -> list[list[str]]:
        """Find all start-to-end paths in the protocol graph.

        Uses NetworkX for path enumeration.
        """
        if self._graph is None or self._protocol is None:
            return []

        start_nodes = [
            n["id"] for n in self._protocol["nodes"]
            if n.get("type") == "start"
        ]
        end_nodes = [
            n["id"] for n in self._protocol["nodes"]
            if n.get("type") == "end"
        ]

        all_paths = []
        for start in start_nodes:
            for end in end_nodes:
                try:
                    paths = list(
                        nx.all_simple_paths(self._graph, start, end, cutoff=50)
                    )
                    all_paths.extend(paths)
                except nx.NetworkXError:
                    continue

        return all_paths

    def check_protocol_validity(self) -> dict:
        """Run structural validity checks on the protocol graph.

        Checks:
        1. All nodes reachable from start
        2. All paths lead to an end node
        3. Decision nodes have ≥2 outgoing edges
        4. No cycles (optional — some protocols have loops)
        """
        if self._graph is None or self._protocol is None:
            return {"error": "No protocol loaded"}

        issues = []

        # 1. Reachability from start
        start_nodes = [
            n["id"] for n in self._protocol["nodes"]
            if n.get("type") == "start"
        ]
        all_node_ids = {n["id"] for n in self._protocol["nodes"]}

        reachable = set()
        for start in start_nodes:
            reachable.update(nx.descendants(self._graph, start))
            reachable.add(start)

        unreachable = all_node_ids - reachable
        if unreachable:
            issues.append(f"Unreachable nodes: {sorted(unreachable)}")

        # 2. Decision nodes with < 2 outgoing
        for node in self._protocol["nodes"]:
            if node.get("type") == "decision":
                out_degree = self._graph.out_degree(node["id"])
                if out_degree < 2:
                    issues.append(
                        f"Decision node {node['id']} has {out_degree} outgoing edges (need ≥2)"
                    )

        # 3. Dangling nodes (no outgoing, not end)
        end_ids = {
            n["id"] for n in self._protocol["nodes"]
            if n.get("type") == "end"
        }
        for node_id in all_node_ids:
            if node_id not in end_ids and self._graph.out_degree(node_id) == 0:
                issues.append(f"Dangling node {node_id} (no outgoing edges, not an end node)")

        # 4. Check for cycles
        has_cycles = not nx.is_directed_acyclic_graph(self._graph)

        return {
            "protocol_id": self._protocol.get("protocol_id", ""),
            "is_valid": len(issues) == 0,
            "has_cycles": has_cycles,
            "issues": issues,
            "num_paths": len(self.find_critical_paths()),
        }


def validate_protocols_with_pyreason(
    directory: str,
    num_timesteps: int = 15,
) -> list[dict]:
    """Validate all extracted protocols using PyReason temporal reasoning.

    Returns list of validation results per protocol.
    """
    directory = Path(directory)
    results = []

    reasoner = ProtocolReasoner()

    for json_file in sorted(directory.glob("*.json")):
        if json_file.name in ("manifest.json",):
            continue

        try:
            reasoner.load_protocol(json_file)

            # Structural validity
            validity = reasoner.check_protocol_validity()

            # Temporal reasoning
            reasoning = reasoner.reason(num_timesteps=num_timesteps)

            results.append({
                "file": json_file.name,
                "validity": validity,
                "reasoning": reasoning,
            })

            status = "VALID" if validity["is_valid"] else f"INVALID ({len(validity['issues'])} issues)"
            complete = "COMPLETE" if reasoning["completion"] else "INCOMPLETE"
            print(f"  {json_file.name}: {status}, {complete}, "
                  f"{reasoning['reachable_nodes']}/{reasoning['total_nodes']} reachable")

        except Exception as e:
            results.append({"file": json_file.name, "error": str(e)})
            print(f"  {json_file.name}: ERROR - {e}")

    return results
