"""
Evaluate extracted protocol graphs against gold standard annotations.

Computes:
- Node-level precision, recall, F1 (text similarity matching)
- Edge-level precision, recall, F1
- Structural validity checks
"""

import sys
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, ".")


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    """Compute similarity between two text strings (0-1)."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def match_nodes(
    gold_nodes: list[dict], extracted_nodes: list[dict], threshold: float = 0.5
) -> tuple[list[tuple], list[dict], list[dict]]:
    """Match extracted nodes to gold nodes using text similarity.

    Returns:
        (matched_pairs, unmatched_gold, unmatched_extracted)
    """
    matched = []
    used_gold = set()
    used_extracted = set()

    # Build similarity matrix and greedily match best pairs
    scores = []
    for i, gn in enumerate(gold_nodes):
        for j, en in enumerate(extracted_nodes):
            sim = text_similarity(gn["text"], en["text"])
            if sim >= threshold:
                scores.append((sim, i, j))

    # Sort by similarity descending, greedily assign
    scores.sort(key=lambda x: x[0], reverse=True)
    for sim, gi, ei in scores:
        if gi not in used_gold and ei not in used_extracted:
            matched.append((gold_nodes[gi], extracted_nodes[ei], sim))
            used_gold.add(gi)
            used_extracted.add(ei)

    unmatched_gold = [gn for i, gn in enumerate(gold_nodes) if i not in used_gold]
    unmatched_extracted = [en for i, en in enumerate(extracted_nodes) if i not in used_extracted]

    return matched, unmatched_gold, unmatched_extracted


def evaluate_nodes(
    gold: dict, extracted: dict, threshold: float = 0.5
) -> dict:
    """Evaluate node extraction quality."""
    gold_nodes = gold["nodes"]
    ext_nodes = extracted["nodes"]

    matched, unmatched_gold, unmatched_ext = match_nodes(
        gold_nodes, ext_nodes, threshold
    )

    tp = len(matched)
    fp = len(unmatched_ext)
    fn = len(unmatched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_sim = sum(m[2] for m in matched) / len(matched) if matched else 0

    # Type accuracy: of matched nodes, how many have the correct type?
    type_correct = sum(
        1 for gn, en, _ in matched if gn["type"] == en["type"]
    )
    type_accuracy = type_correct / len(matched) if matched else 0

    return {
        "gold_count": len(gold_nodes),
        "extracted_count": len(ext_nodes),
        "matched": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_text_similarity": round(avg_sim, 3),
        "type_accuracy": round(type_accuracy, 3),
        "unmatched_gold": [n["text"][:60] for n in unmatched_gold],
        "unmatched_extracted": [n["text"][:60] for n in unmatched_ext],
    }


def evaluate_edges(
    gold: dict, extracted: dict, node_threshold: float = 0.5
) -> dict:
    """Evaluate edge extraction quality.

    Computes:
    - Edge connectivity F1: based on (from, to) pairs
    - Edge condition accuracy: for matched edges, similarity of condition labels
    """
    gold_nodes = gold["nodes"]
    ext_nodes = extracted["nodes"]

    # First match nodes to create an ID mapping
    matched, _, _ = match_nodes(gold_nodes, ext_nodes, node_threshold)

    # Build mapping: gold_node_id -> extracted_node_id
    id_map = {}
    for gn, en, _ in matched:
        id_map[gn["id"]] = en["id"]

    # Convert gold edges to (from_ext_id, to_ext_id) pairs using mapping
    gold_edges_mapped = set()
    gold_edge_conditions = {}  # (from, to) -> condition
    for edge in gold["edges"]:
        from_id = id_map.get(edge["from"])
        to_id = id_map.get(edge["to"])
        if from_id and to_id:
            gold_edges_mapped.add((from_id, to_id))
            gold_edge_conditions[(from_id, to_id)] = edge.get("condition", "")

    # Extracted edges as (from, to) pairs + conditions
    ext_edges = {(e["from"], e["to"]) for e in extracted["edges"]}
    ext_edge_conditions = {
        (e["from"], e["to"]): e.get("condition", "") for e in extracted["edges"]
    }

    tp = len(gold_edges_mapped & ext_edges)
    fp = len(ext_edges - gold_edges_mapped)
    fn = len(gold_edges_mapped - ext_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Condition accuracy: for matched edges, compare condition labels
    matched_edges = gold_edges_mapped & ext_edges
    condition_scores = []
    for edge_key in matched_edges:
        gold_cond = gold_edge_conditions.get(edge_key, "")
        ext_cond = ext_edge_conditions.get(edge_key, "")
        if gold_cond or ext_cond:
            sim = text_similarity(gold_cond, ext_cond) if (gold_cond and ext_cond) else 0.0
            condition_scores.append(sim)
        # If both are empty, skip — no condition to compare

    avg_condition_sim = (
        sum(condition_scores) / len(condition_scores)
        if condition_scores
        else 1.0  # All matched edges have no conditions = perfect
    )
    condition_exact_match = (
        sum(1 for s in condition_scores if s >= 0.8) / len(condition_scores)
        if condition_scores
        else 1.0
    )

    return {
        "gold_edges": len(gold["edges"]),
        "extracted_edges": len(extracted["edges"]),
        "mappable_gold_edges": len(gold_edges_mapped),
        "matched": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "condition_similarity": round(avg_condition_sim, 3),
        "condition_exact_match_rate": round(condition_exact_match, 3),
        "conditions_evaluated": len(condition_scores),
    }


def structural_checks(extracted: dict) -> dict:
    """Run structural validity checks on an extracted graph."""
    nodes = extracted.get("nodes", [])
    edges = extracted.get("edges", [])
    node_ids = {n["id"] for n in nodes}

    issues = []

    # Check start/end nodes exist
    starts = [n for n in nodes if n["type"] == "start"]
    ends = [n for n in nodes if n["type"] == "end"]
    if len(starts) == 0:
        issues.append("No start node")
    if len(ends) == 0:
        issues.append("No end node")

    # Check for dangling edge references
    for e in edges:
        if e["from"] not in node_ids:
            issues.append(f"Dangling edge from: {e['from']}")
        if e["to"] not in node_ids:
            issues.append(f"Dangling edge to: {e['to']}")

    # Check orphan nodes
    connected = set()
    for e in edges:
        connected.add(e["from"])
        connected.add(e["to"])
    orphans = node_ids - connected
    if orphans:
        issues.append(f"Orphan nodes: {orphans}")

    # Check decision nodes have >=2 outgoing edges
    decisions = [n for n in nodes if n["type"] == "decision"]
    for d in decisions:
        outgoing = sum(1 for e in edges if e["from"] == d["id"])
        if outgoing < 2:
            issues.append(f"Decision {d['id']} has {outgoing} outgoing edge(s)")

    return {
        "valid": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
    }


def full_evaluation(gold: dict, extracted: dict) -> dict:
    """Run full evaluation of extracted vs gold protocol."""
    return {
        "protocol": gold.get("title", "?"),
        "nodes": evaluate_nodes(gold, extracted),
        "edges": evaluate_edges(gold, extracted),
        "structure": structural_checks(extracted),
        "graph": evaluate_graph_level(gold, extracted),
    }


def evaluate_graph_level(
    gold: dict, extracted: dict, node_threshold: float = 0.5
) -> dict:
    """Graph-level evaluation metrics.

    Computes:
    - Branching accuracy: % of decision nodes with correct outgoing edge count
    - Path count comparison: number of start→end paths in each graph
    """
    gold_nodes = gold["nodes"]
    ext_nodes = extracted["nodes"]

    matched, _, _ = match_nodes(gold_nodes, ext_nodes, node_threshold)
    id_map = {gn["id"]: en["id"] for gn, en, _ in matched}

    # Branching accuracy: for matched decision nodes, check outgoing edge count
    gold_edges = gold["edges"]
    ext_edges = extracted["edges"]

    decision_matches = [
        (gn, en) for gn, en, _ in matched if gn["type"] == "decision"
    ]
    branching_correct = 0
    branching_total = len(decision_matches)

    for gn, en in decision_matches:
        gold_outgoing = sum(1 for e in gold_edges if e["from"] == gn["id"])
        ext_outgoing = sum(1 for e in ext_edges if e["from"] == en["id"])
        if gold_outgoing == ext_outgoing:
            branching_correct += 1

    branching_accuracy = (
        branching_correct / branching_total if branching_total > 0 else 1.0
    )

    # Path count: count start→end paths in each graph
    gold_path_count = _count_paths(gold)
    ext_path_count = _count_paths(extracted)

    return {
        "branching_accuracy": round(branching_accuracy, 3),
        "branching_correct": branching_correct,
        "branching_total": branching_total,
        "gold_paths": gold_path_count,
        "extracted_paths": ext_path_count,
    }


def _count_paths(graph: dict, max_depth: int = 50) -> int:
    """Count the number of distinct start→end paths in a protocol graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    start_ids = [n["id"] for n in nodes if n["type"] == "start"]
    end_ids = {n["id"] for n in nodes if n["type"] == "end"}

    # Build adjacency list
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    path_count = 0

    def dfs(node_id: str, depth: int) -> None:
        nonlocal path_count
        if depth > max_depth:
            return
        if node_id in end_ids:
            path_count += 1
            return
        for neighbor in adj.get(node_id, []):
            dfs(neighbor, depth + 1)

    for start in start_ids:
        dfs(start, 0)

    return path_count


# --- Main ---
if __name__ == "__main__":
    pairs = [
        ("evaluation/gold/risk_management_01.json", "evaluation/extracted/risk_management_process.json"),
        ("evaluation/gold/patient_assessment_01.json", "evaluation/extracted/patient_assessment.json"),
        # triage_01 is now a few-shot example; compare against fresh text extraction
        ("evaluation/gold/triage_01.json", "evaluation/extracted/multi_casualty_triage_system.json"),
    ]

    all_results = []

    for gold_path, ext_path in pairs:
        if not Path(gold_path).exists() or not Path(ext_path).exists():
            print(f"SKIP: {gold_path} or {ext_path} not found")
            continue

        with open(gold_path) as f:
            gold = json.load(f)
        with open(ext_path) as f:
            extracted = json.load(f)

        result = full_evaluation(gold, extracted)
        result["source"] = Path(ext_path).stem
        all_results.append(result)

        print(f"\n{'='*60}")
        print(f"Protocol: {result['protocol']} [{result['source']}]")
        print(f"{'='*60}")
        n = result["nodes"]
        print(f"NODES  gold={n['gold_count']} ext={n['extracted_count']} "
              f"matched={n['matched']}")
        print(f"       P={n['precision']} R={n['recall']} F1={n['f1']} "
              f"type_acc={n['type_accuracy']}")
        if n["unmatched_gold"]:
            print(f"       Missing from extraction: {n['unmatched_gold']}")
        if n["unmatched_extracted"]:
            print(f"       Extra in extraction: {n['unmatched_extracted']}")

        e = result["edges"]
        print(f"EDGES  gold={e['gold_edges']} ext={e['extracted_edges']} "
              f"matched={e['matched']}")
        print(f"       P={e['precision']} R={e['recall']} F1={e['f1']}")
        print(f"       Condition sim={e['condition_similarity']} "
              f"exact_match={e['condition_exact_match_rate']} "
              f"(n={e['conditions_evaluated']})")

        s = result["structure"]
        if s["valid"]:
            print(f"STRUCT valid=True")
        else:
            print(f"STRUCT valid=False issues={s['issues']}")

        g = result["graph"]
        print(f"GRAPH  branching_acc={g['branching_accuracy']} "
              f"({g['branching_correct']}/{g['branching_total']}) "
              f"paths: gold={g['gold_paths']} ext={g['extracted_paths']}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        avg_node_f1 = sum(r["nodes"]["f1"] for r in all_results) / len(all_results)
        avg_edge_f1 = sum(r["edges"]["f1"] for r in all_results) / len(all_results)
        avg_cond_sim = sum(r["edges"]["condition_similarity"] for r in all_results) / len(all_results)
        avg_branch_acc = sum(r["graph"]["branching_accuracy"] for r in all_results) / len(all_results)
        all_valid = all(r["structure"]["valid"] for r in all_results)
        print(f"Avg Node F1: {avg_node_f1:.3f}")
        print(f"Avg Edge F1: {avg_edge_f1:.3f}")
        print(f"Avg Condition Similarity: {avg_cond_sim:.3f}")
        print(f"Avg Branching Accuracy: {avg_branch_acc:.3f}")
        print(f"All structurally valid: {all_valid}")
