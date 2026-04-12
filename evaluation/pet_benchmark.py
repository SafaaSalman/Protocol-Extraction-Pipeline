"""Benchmark extraction pipeline against PET dataset.

Runs our extraction pipeline on PET source texts, compares against PET gold
standards (converted via pet_adapter), and reports metrics comparable to
published PET baselines.

Published baselines (Neuberger et al. 2024, arXiv:2407.18540):
  Using GPT-4o on PET dataset (45 documents):
    Mention Detection (MD) 3-shot:  P=0.72  R=0.77  F1=0.74
    Entity Resolution (ER) 3-shot:  P=0.79  R=0.70  F1=0.74
    Relation Extraction (RE) 3-shot: P=0.90  R=0.89  F1=0.89
    RE zero-shot:                   P=0.88  R=0.85  F1=0.86

  ML baseline (Neuberger et al. 2023, CoopIS):
    MD: P=0.73  R=0.64  F1=0.69
    ER: P=0.55  R=0.51  F1=0.52
    RE: P=0.79  R=0.66  F1=0.72

Our evaluation methodology:
  - Use token_overlap_similarity for node matching (PET gold has short
    verb spans like "receives" while our extraction produces full
    descriptions like "Receive the customer order")
  - Activity detection  = matching action nodes (PET Activity entities)
  - Gateway detection   = matching decision nodes (PET XOR Gateway entities)
  - Flow detection      = edge matching after node alignment
"""

import sys
import json
import time

sys.path.insert(0, ".")

from src.graph.pet_adapter import PETAdapter
from src.extraction.protocol_extractor import extract_protocol_from_text, review_edges
from evaluate import (
    evaluate_nodes, evaluate_edges, evaluate_graph_level,
    match_nodes, token_overlap_similarity, normalize_text,
)


def compute_per_type_f1(gold_nodes, ext_nodes, node_type, similarity_fn):
    """Compute F1 for a specific node type (activity/gateway detection)."""
    gold_typed = [n for n in gold_nodes if n["type"] == node_type]
    ext_typed = [n for n in ext_nodes if n["type"] == node_type]

    if not gold_typed and not ext_typed:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "gold": 0, "ext": 0}

    matched, unmatched_gold, unmatched_ext = match_nodes(
        gold_typed, ext_typed, threshold=0.3, similarity_fn=similarity_fn,
    )

    tp = len(matched)
    fp = len(unmatched_ext)
    fn = len(unmatched_gold)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "gold": len(gold_typed), "ext": len(ext_typed), "matched": tp,
    }


def run_pet_benchmark(num_samples: int = 10, use_edge_review: bool = True):
    """Run pipeline on PET documents and evaluate."""

    sim_fn = token_overlap_similarity

    # 1. Load PET data and convert to gold standards
    adapter = PETAdapter()
    adapter.download()
    raw_data = adapter._dataset
    gold_protocols = adapter.convert_all()

    # Only take first num_samples
    raw_data = raw_data[:num_samples]
    gold_protocols = gold_protocols[:num_samples]

    results = []
    total_tokens = 0
    total_time = 0

    for i, (doc, gold) in enumerate(zip(raw_data, gold_protocols)):
        doc_name = doc["document name"]
        source_text = " ".join(doc["tokens"])

        print(f"\n[{i+1}/{num_samples}] {doc_name} ({len(doc['tokens'])} tokens)")

        t0 = time.time()
        try:
            extracted = extract_protocol_from_text(
                section_text=source_text,
                title=doc_name,
                pages=[1],
                source_pdf="PET_dataset",
                model="gpt-4o",
                use_few_shot=True,
                format_hint="unknown",
            )

            if use_edge_review:
                extracted = review_edges(extracted, source_text, model="gpt-4o-mini")

            elapsed = time.time() - t0
            total_time += elapsed

            tokens_used = extracted.get("_extraction_meta", {}).get("total_tokens", 0)
            total_tokens += tokens_used

            # Evaluate with token-overlap matching
            node_result = evaluate_nodes(gold, extracted, threshold=0.3,
                                         similarity_fn=sim_fn)
            edge_result = evaluate_edges(gold, extracted, node_threshold=0.3,
                                         similarity_fn=sim_fn)
            graph_result = evaluate_graph_level(gold, extracted, node_threshold=0.3,
                                                similarity_fn=sim_fn)

            # Per-type metrics (comparable to PET baselines)
            activity_f1 = compute_per_type_f1(
                gold["nodes"], extracted["nodes"], "action", sim_fn)
            gateway_f1 = compute_per_type_f1(
                gold["nodes"], extracted["nodes"], "decision", sim_fn)

            result = {
                "document": doc_name,
                "gold_nodes": len(gold["nodes"]),
                "ext_nodes": len(extracted["nodes"]),
                "gold_edges": len(gold["edges"]),
                "ext_edges": len(extracted["edges"]),
                # Overall metrics
                "node_f1": node_result["f1"],
                "node_precision": node_result["precision"],
                "node_recall": node_result["recall"],
                "type_accuracy": node_result["type_accuracy"],
                "edge_f1": edge_result["f1"],
                "edge_precision": edge_result["precision"],
                "edge_recall": edge_result["recall"],
                "cond_similarity": edge_result["condition_similarity"],
                "branching_acc": graph_result["branching_accuracy"],
                # PET-comparable per-type metrics
                "activity_f1": activity_f1["f1"],
                "activity_p": activity_f1["precision"],
                "activity_r": activity_f1["recall"],
                "gateway_f1": gateway_f1["f1"],
                "gateway_p": gateway_f1["precision"],
                "gateway_r": gateway_f1["recall"],
                "gold_actions": activity_f1["gold"],
                "ext_actions": activity_f1["ext"],
                "gold_decisions": gateway_f1["gold"],
                "ext_decisions": gateway_f1["ext"],
                "time_s": round(elapsed, 1),
                "tokens": tokens_used,
            }
            results.append(result)

            print(f"  Nodes: {node_result['f1']:.3f} F1  "
                  f"Activity: {activity_f1['f1']:.3f}  "
                  f"Gateway: {gateway_f1['f1']:.3f}")
            print(f"  Edges: {edge_result['f1']:.3f} F1  "
                  f"Type acc: {node_result['type_accuracy']:.3f}  "
                  f"Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            results.append({
                "document": doc_name,
                "error": str(e),
                "time_s": round(elapsed, 1),
            })

    # Aggregate
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*70}")
    print(f"PET BENCHMARK RESULTS ({len(valid)}/{num_samples} successful)")
    print(f"{'='*70}")

    if valid:
        avg = lambda key: sum(r[key] for r in valid) / len(valid)

        print(f"\n  Overall Metrics (avg over {len(valid)} documents):")
        print(f"  {'─'*55}")
        print(f"  Node F1:          {avg('node_f1'):.3f}  "
              f"(P={avg('node_precision'):.3f} R={avg('node_recall'):.3f})")
        print(f"  Edge F1:          {avg('edge_f1'):.3f}  "
              f"(P={avg('edge_precision'):.3f} R={avg('edge_recall'):.3f})")
        print(f"  Type Accuracy:    {avg('type_accuracy'):.3f}")
        print(f"  Cond Similarity:  {avg('cond_similarity'):.3f}")
        print(f"  Branching Acc:    {avg('branching_acc'):.3f}")

        print(f"\n  PET-Comparable Metrics (vs Neuberger 2024 baselines):")
        print(f"  {'─'*55}")
        print(f"  Activity Detection (MD action nodes):")
        print(f"    Ours:      P={avg('activity_p'):.3f}  "
              f"R={avg('activity_r'):.3f}  F1={avg('activity_f1'):.3f}")
        print(f"    GPT-4o*:   P=0.720       R=0.770       F1=0.740")
        print(f"    ML base*:  P=0.730       R=0.640       F1=0.690")
        print(f"  Gateway Detection (MD decision nodes):")
        print(f"    Ours:      P={avg('gateway_p'):.3f}  "
              f"R={avg('gateway_r'):.3f}  F1={avg('gateway_f1'):.3f}")
        print(f"  Flow Detection (RE edges):")
        print(f"    Ours:      P={avg('edge_precision'):.3f}  "
              f"R={avg('edge_recall'):.3f}  F1={avg('edge_f1'):.3f}")
        print(f"    GPT-4o*:   P=0.900       R=0.890       F1=0.890")
        print(f"    ML base*:  P=0.790       R=0.660       F1=0.720")
        print(f"  * Neuberger et al. 2024, 3-shot, full 45-doc PET")

        print(f"\n  Counts:")
        print(f"  {'─'*55}")
        print(f"  Activities:  gold={sum(r['gold_actions'] for r in valid):4d}  "
              f"extracted={sum(r['ext_actions'] for r in valid):4d}")
        print(f"  Gateways:    gold={sum(r['gold_decisions'] for r in valid):4d}  "
              f"extracted={sum(r['ext_decisions'] for r in valid):4d}")

        print(f"\n  Efficiency:")
        print(f"  {'─'*55}")
        print(f"  Total time:       {total_time:.1f}s")
        print(f"  Avg time/doc:     {total_time/len(valid):.1f}s")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Avg tokens/doc:   {total_tokens//len(valid):,}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r['document']}: {r['error']}")

    # Save full results
    out_path = "evaluation/pet_benchmark_results.json"
    summary = {}
    if valid:
        summary = {
            "num_evaluated": len(valid),
            "num_errors": len(errors),
            "avg_node_f1": round(avg('node_f1'), 3),
            "avg_edge_f1": round(avg('edge_f1'), 3),
            "avg_type_accuracy": round(avg('type_accuracy'), 3),
            "avg_activity_f1": round(avg('activity_f1'), 3),
            "avg_gateway_f1": round(avg('gateway_f1'), 3),
            "avg_cond_similarity": round(avg('cond_similarity'), 3),
            "avg_branching_accuracy": round(avg('branching_acc'), 3),
            "total_time_s": round(total_time, 1),
            "total_tokens": total_tokens,
        }

    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "num_samples": num_samples,
                "use_edge_review": use_edge_review,
                "model": "gpt-4o",
                "edge_review_model": "gpt-4o-mini",
                "matching": "token_overlap_similarity",
                "threshold": 0.3,
            },
            "baselines": {
                "neuberger_2024_gpt4o_3shot": {
                    "mention_detection_f1": 0.74,
                    "entity_resolution_f1": 0.74,
                    "relation_extraction_f1": 0.89,
                },
                "neuberger_2023_ml_baseline": {
                    "mention_detection_f1": 0.69,
                    "entity_resolution_f1": 0.52,
                    "relation_extraction_f1": 0.72,
                },
            },
            "summary": summary,
            "per_document": results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark on PET dataset")
    parser.add_argument("-n", "--num-samples", type=int, default=10,
                        help="Number of PET documents to evaluate (default: 10)")
    parser.add_argument("--no-edge-review", action="store_true",
                        help="Skip edge review pass")
    args = parser.parse_args()

    run_pet_benchmark(
        num_samples=args.num_samples,
        use_edge_review=not args.no_edge_review,
    )
