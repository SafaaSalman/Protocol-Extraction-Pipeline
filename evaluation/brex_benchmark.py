"""Benchmark extraction pipeline against BREX dataset.

Runs our extraction pipeline on BREX source texts (Chinese business rules),
compares against BREX gold standards (converted via brex_adapter), and reports
metrics comparable to published BREX baselines.

Published baselines (Yang et al. 2025, arXiv:2505.18542):
  Rule NER F1 (best per model):
    DeepSeek-R1:     F1=0.90
    Gemini-2.5-Pro:  F1=0.88
    GPT-o3-mini:     F1=0.87
    GPT-4o:          F1=0.85
    Claude-3.7:      F1=0.84

  Dependency Extraction F1 (best per model):
    Gemini-2.5-Pro:  F1=0.76
    DeepSeek-R1:     F1=0.72
    GPT-o3-mini:     F1=0.67
    GPT-4o:          F1=0.53
    Claude-3.7:      F1=0.51

Our evaluation methodology:
  - BREX gold is converted to protocol graph format via brex_adapter
  - Source text (Chinese) is passed directly to GPT-4o for extraction
  - Node matching uses token_overlap_similarity
  - Rule NER ≈ node matching (action + decision nodes)
  - Dependency ≈ edge matching after node alignment
  - Cross-lingual: our system prompt is English, input is Chinese
"""

import sys
import json
import time
import argparse

import os
sys.path.insert(0, ".")

# Load API key from env or .env file
if not os.environ.get("OPENAI_API_KEY"):
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY"):
                    key = line.strip().split("=", 1)[1].strip().strip('"')
                    os.environ["OPENAI_API_KEY"] = key

from src.graph.brex_adapter import BREXAdapter
from src.extraction.protocol_extractor import extract_protocol_from_text, review_edges
from evaluate import (
    evaluate_nodes, evaluate_edges, evaluate_graph_level,
    match_nodes, token_overlap_similarity,
)


def compute_per_type_f1(gold_nodes, ext_nodes, node_type, similarity_fn):
    """Compute F1 for a specific node type."""
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


def run_brex_benchmark(num_samples: int = 10, use_edge_review: bool = True):
    """Run pipeline on BREX documents and evaluate."""

    sim_fn = token_overlap_similarity

    # 1. Load BREX data and convert to gold standards
    adapter = BREXAdapter()
    adapter.download()
    raw_data = adapter._dataset[:num_samples]
    gold_protocols = adapter.convert_all()[:num_samples]

    results = []
    total_tokens = 0
    total_time = 0

    for i, (doc, gold) in enumerate(zip(raw_data, gold_protocols)):
        domain = doc.get("\u9886\u57df", "unknown")  # 领域
        intent = doc.get("\u610f\u56fe", "unknown")  # 意图
        source_text = doc.get("\u6587\u672c", "")     # 文本
        doc_name = f"{domain}_{intent}"

        print(f"\n[{i+1}/{num_samples}] {doc_name} "
              f"({len(source_text)} chars, {len(gold['nodes'])} gold nodes)")

        t0 = time.time()
        try:
            extracted = extract_protocol_from_text(
                section_text=source_text,
                title=intent,
                pages=[1],
                source_pdf="BREX_dataset",
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

            # Per-type metrics
            action_f1 = compute_per_type_f1(
                gold["nodes"], extracted["nodes"], "action", sim_fn)
            decision_f1 = compute_per_type_f1(
                gold["nodes"], extracted["nodes"], "decision", sim_fn)

            result = {
                "document": doc_name,
                "domain": domain,
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
                # Per-type metrics
                "action_f1": action_f1["f1"],
                "action_p": action_f1["precision"],
                "action_r": action_f1["recall"],
                "decision_f1": decision_f1["f1"],
                "decision_p": decision_f1["precision"],
                "decision_r": decision_f1["recall"],
                "gold_actions": action_f1["gold"],
                "ext_actions": action_f1["ext"],
                "gold_decisions": decision_f1["gold"],
                "ext_decisions": decision_f1["ext"],
                "time_s": round(elapsed, 1),
                "tokens": tokens_used,
            }
            results.append(result)

            print(f"  Nodes: {node_result['f1']:.3f} F1  "
                  f"Actions: {action_f1['f1']:.3f}  "
                  f"Decisions: {decision_f1['f1']:.3f}")
            print(f"  Edges: {edge_result['f1']:.3f} F1  "
                  f"Type acc: {node_result['type_accuracy']:.3f}  "
                  f"Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "document": doc_name,
                "domain": domain,
                "error": str(e),
                "time_s": round(elapsed, 1),
            })

    # Aggregate
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*70}")
    print(f"BREX BENCHMARK RESULTS ({len(valid)}/{num_samples} successful)")
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

        print(f"\n  BREX-Comparable Metrics (vs Yang et al. 2025 baselines):")
        print(f"  {'─'*55}")
        print(f"  Rule NER (node matching):")
        print(f"    Ours:           P={avg('node_precision'):.3f}  "
              f"R={avg('node_recall'):.3f}  F1={avg('node_f1'):.3f}")
        print(f"    GPT-4o*:        F1=0.850")
        print(f"    DeepSeek-R1*:   F1=0.900")
        print(f"    Gemini-2.5*:    F1=0.880")
        print(f"  Dependency Extraction (edge matching):")
        print(f"    Ours:           P={avg('edge_precision'):.3f}  "
              f"R={avg('edge_recall'):.3f}  F1={avg('edge_f1'):.3f}")
        print(f"    GPT-4o*:        F1=0.530")
        print(f"    DeepSeek-R1*:   F1=0.720")
        print(f"    Gemini-2.5*:    F1=0.760")
        print(f"  * Yang et al. 2025, full 409-doc BREX dataset")

        print(f"\n  Per-Type Breakdown:")
        print(f"  {'─'*55}")
        print(f"  Action nodes:   P={avg('action_p'):.3f}  "
              f"R={avg('action_r'):.3f}  F1={avg('action_f1'):.3f}  "
              f"(gold={sum(r['gold_actions'] for r in valid)}, "
              f"ext={sum(r['ext_actions'] for r in valid)})")
        print(f"  Decision nodes: P={avg('decision_p'):.3f}  "
              f"R={avg('decision_r'):.3f}  F1={avg('decision_f1'):.3f}  "
              f"(gold={sum(r['gold_decisions'] for r in valid)}, "
              f"ext={sum(r['ext_decisions'] for r in valid)})")

        # Per-domain aggregation
        domains = {}
        for r in valid:
            d = r["domain"]
            if d not in domains:
                domains[d] = []
            domains[d].append(r)

        if len(domains) > 1:
            print(f"\n  Per-Domain Performance:")
            print(f"  {'─'*55}")
            print(f"  {'Domain':<20} {'Docs':>4}  {'Node F1':>7}  {'Edge F1':>7}")
            for d in sorted(domains.keys(),
                            key=lambda x: -len(domains[x])):
                dd = domains[d]
                nf1 = sum(r['node_f1'] for r in dd) / len(dd)
                ef1 = sum(r['edge_f1'] for r in dd) / len(dd)
                print(f"  {d:<20} {len(dd):4d}  {nf1:7.3f}  {ef1:7.3f}")

        print(f"\n  Efficiency:")
        print(f"  {'─'*55}")
        print(f"  Total time:       {total_time:.1f}s")
        print(f"  Avg time/doc:     {total_time/len(valid):.1f}s")
        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Avg tokens/doc:   {total_tokens//len(valid):,}")

        print(f"\n  Note: Cross-lingual evaluation (Chinese text, English system prompt)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r['document']}: {r['error']}")

    # Save full results
    out_path = "evaluation/brex_benchmark_results.json"
    summary = {}
    if valid:
        summary = {
            "num_evaluated": len(valid),
            "num_errors": len(errors),
            "avg_node_f1": round(avg('node_f1'), 3),
            "avg_edge_f1": round(avg('edge_f1'), 3),
            "avg_type_accuracy": round(avg('type_accuracy'), 3),
            "avg_action_f1": round(avg('action_f1'), 3),
            "avg_decision_f1": round(avg('decision_f1'), 3),
            "avg_cond_similarity": round(avg('cond_similarity'), 3),
            "avg_branching_accuracy": round(avg('branching_acc'), 3),
            "total_time_s": round(total_time, 1),
            "total_tokens": total_tokens,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "num_samples": num_samples,
                "use_edge_review": use_edge_review,
                "model": "gpt-4o",
                "edge_review_model": "gpt-4o-mini",
                "matching": "token_overlap_similarity",
                "threshold": 0.3,
                "note": "Cross-lingual: Chinese source text, English system prompt",
            },
            "baselines": {
                "yang_2025_gpt4o": {
                    "rule_ner_f1": 0.85,
                    "dependency_f1": 0.53,
                },
                "yang_2025_deepseek_r1": {
                    "rule_ner_f1": 0.90,
                    "dependency_f1": 0.72,
                },
                "yang_2025_gemini_25_pro": {
                    "rule_ner_f1": 0.88,
                    "dependency_f1": 0.76,
                },
                "yang_2025_claude_37": {
                    "rule_ner_f1": 0.84,
                    "dependency_f1": 0.51,
                },
            },
            "summary": summary,
            "per_document": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark on BREX dataset")
    parser.add_argument("-n", "--num-samples", type=int, default=10,
                        help="Number of BREX documents to evaluate (default: 10)")
    parser.add_argument("--no-edge-review", action="store_true",
                        help="Skip edge review pass")
    args = parser.parse_args()

    run_brex_benchmark(
        num_samples=args.num_samples,
        use_edge_review=not args.no_edge_review,
    )
