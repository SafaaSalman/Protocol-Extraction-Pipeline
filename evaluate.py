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
    """Evaluate edge extraction quality."""
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
    for edge in gold["edges"]:
        from_id = id_map.get(edge["from"])
        to_id = id_map.get(edge["to"])
        if from_id and to_id:
            gold_edges_mapped.add((from_id, to_id))

    # Extracted edges as (from, to) pairs
    ext_edges = {(e["from"], e["to"]) for e in extracted["edges"]}

    tp = len(gold_edges_mapped & ext_edges)
    fp = len(ext_edges - gold_edges_mapped)
    fn = len(gold_edges_mapped - ext_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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
    }


# --- Main ---
if __name__ == "__main__":
    pairs = [
        ("evaluation/gold/risk_management_01.json", "evaluation/extracted/risk_management_process.json"),
        ("evaluation/gold/patient_assessment_01.json", "evaluation/extracted/patient_assessment.json"),
        ("evaluation/gold/triage_01.json", "evaluation/extracted/multi_casualty_triage_system.json"),
        ("evaluation/gold/triage_01.json", "evaluation/extracted/triage_multimodal.json"),
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

        s = result["structure"]
        if s["valid"]:
            print(f"STRUCT valid=True")
        else:
            print(f"STRUCT valid=False issues={s['issues']}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        avg_node_f1 = sum(r["nodes"]["f1"] for r in all_results) / len(all_results)
        avg_edge_f1 = sum(r["edges"]["f1"] for r in all_results) / len(all_results)
        all_valid = all(r["structure"]["valid"] for r in all_results)
        print(f"Avg Node F1: {avg_node_f1:.3f}")
        print(f"Avg Edge F1: {avg_edge_f1:.3f}")
        print(f"All structurally valid: {all_valid}")
