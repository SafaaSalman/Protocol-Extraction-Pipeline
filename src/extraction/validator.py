"""
Protocol Graph Validator

Two-stage validation for extracted protocol graphs:
1. Structural validation (programmatic, no LLM)
2. Semantic validation (LLM reviews graph against source text)

The semantic validator checks:
- Completeness: Are steps from the source text missing?
- Accuracy: Does each node match the source evidence?
- Ordering: Are sequential steps in the correct order?
- Branching: Are decision paths modeled correctly?
- Redundancy: Are there duplicate or overly granular nodes?

Returns a validated (possibly corrected) graph with an audit trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


# ── Structural validation (free, no LLM) ────────────────────────────

@dataclass
class StructuralIssue:
    severity: str   # "error" | "warning"
    code: str       # machine-readable code
    message: str    # human-readable description
    node_id: Optional[str] = None


def validate_structure(graph: dict) -> list[StructuralIssue]:
    """Run programmatic structural checks on a protocol graph.

    These are deterministic rules — no LLM needed.
    """
    issues: list[StructuralIssue] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = {n["id"] for n in nodes}

    # 1. Must have start and end nodes
    starts = [n for n in nodes if n["type"] == "start"]
    ends = [n for n in nodes if n["type"] == "end"]
    if len(starts) == 0:
        issues.append(StructuralIssue("error", "no_start", "No start node found"))
    if len(starts) > 1:
        ids = [n["id"] for n in starts]
        issues.append(StructuralIssue("warning", "multi_start",
                                       f"Multiple start nodes: {ids}"))
    if len(ends) == 0:
        issues.append(StructuralIssue("error", "no_end", "No end node found"))

    # 2. Dangling edge references
    for e in edges:
        if e["from"] not in node_ids:
            issues.append(StructuralIssue("error", "dangling_from",
                                           f"Edge references unknown node: {e['from']}",
                                           node_id=e["from"]))
        if e["to"] not in node_ids:
            issues.append(StructuralIssue("error", "dangling_to",
                                           f"Edge references unknown node: {e['to']}",
                                           node_id=e["to"]))

    # 3. Orphan nodes (not connected to any edge)
    connected = set()
    for e in edges:
        connected.add(e["from"])
        connected.add(e["to"])
    for n in nodes:
        if n["id"] not in connected:
            issues.append(StructuralIssue("warning", "orphan",
                                           f"Orphan node not connected to any edge: {n['id']}",
                                           node_id=n["id"]))

    # 4. Decision nodes must have >=2 outgoing edges with conditions
    for n in nodes:
        if n["type"] == "decision":
            outgoing = [e for e in edges if e["from"] == n["id"]]
            if len(outgoing) < 2:
                issues.append(StructuralIssue("error", "decision_branching",
                                               f"Decision node {n['id']} has {len(outgoing)} "
                                               f"outgoing edge(s), need >=2",
                                               node_id=n["id"]))
            # Check that outgoing edges have conditions
            missing_cond = [e for e in outgoing if not e.get("condition")]
            if missing_cond and len(outgoing) >= 2:
                issues.append(StructuralIssue("warning", "missing_condition",
                                               f"Decision node {n['id']} has edges without conditions",
                                               node_id=n["id"]))

    # 5. Start node should have no incoming edges
    for s in starts:
        incoming = [e for e in edges if e["to"] == s["id"]]
        if incoming:
            issues.append(StructuralIssue("warning", "start_has_incoming",
                                           f"Start node {s['id']} has incoming edges"))

    # 6. End nodes should have no outgoing edges
    for en in ends:
        outgoing = [e for e in edges if e["from"] == en["id"]]
        if outgoing:
            issues.append(StructuralIssue("warning", "end_has_outgoing",
                                           f"End node {en['id']} has outgoing edges"))

    # 7. Node IDs should be sequential (n1, n2, ...)
    expected_ids = {f"n{i}" for i in range(1, len(nodes) + 1)}
    actual_ids = {n["id"] for n in nodes}
    if actual_ids != expected_ids:
        issues.append(StructuralIssue("warning", "non_sequential_ids",
                                       f"Node IDs are not sequential n1..n{len(nodes)}"))

    # 8. Duplicate edges
    edge_keys = [(e["from"], e["to"], e.get("condition", "")) for e in edges]
    seen = set()
    for key in edge_keys:
        if key in seen:
            issues.append(StructuralIssue("warning", "duplicate_edge",
                                           f"Duplicate edge: {key[0]} -> {key[1]}"))
        seen.add(key)

    return issues


# ── Semantic validation (LLM) ───────────────────────────────────────

VALIDATOR_SYSTEM_PROMPT = """\
You are an expert reviewer of protocol graph extractions from emergency management documents.

You are given:
1. The source text from a PDF section
2. A protocol graph (JSON) extracted from that text

Your job is to review the extraction for quality and correctness, then return a corrected version if needed.

## Review Checklist

1. **Completeness**: Are there steps, decisions, or actions in the source text that are NOT captured in the graph?
2. **Accuracy**: Does each node's text faithfully represent the source? Are there fabricated steps not in the source?
3. **Ordering**: For sequential protocols, are the steps in the correct order as presented in the source?
4. **Branching**: For decision points, are all branches captured? Are conditions correct?
5. **Granularity**: Are nodes at the right level — not too coarse (merging distinct steps) or too fine (splitting one step into many)?
6. **Node types**: Is each node correctly typed? (start/end/action/decision)
7. **Edge conditions**: Do decision edges have clear, accurate conditions from the source text?
8. **Structural**: Every decision has >=2 outgoing edges. Exactly one start node. At least one end node. No orphans.

## Output Format

Return a JSON object with:
{
  "review_passed": true/false,
  "issues": [
    {
      "type": "missing_step|fabricated_step|wrong_order|missing_branch|wrong_type|wrong_condition|merged_steps|split_steps|other",
      "severity": "error|warning",
      "description": "<clear description of the issue>",
      "affected_nodes": ["n1", "n2"]
    }
  ],
  "corrections_made": true/false,
  "correction_summary": "<brief description of what was changed, or null if no changes>",
  "corrected_graph": { ... the full corrected graph JSON, or null if no corrections needed ... }
}

## Rules

- If the extraction is good, set review_passed=true, issues=[], corrections_made=false, corrected_graph=null.
- If you find issues, fix them in corrected_graph. Keep the same schema structure.
- Do NOT add steps that aren't in the source text. Only add what was missed from the source.
- Preserve the original node ID scheme (n1, n2, ...) but renumber if nodes are added/removed.
- Keep all evidence references from the original where applicable.
- Return ONLY valid JSON. No markdown, no explanation, no code fences."""


def validate_semantic(
    graph: dict,
    section_text: str,
    model: str = "gpt-4o",
) -> dict:
    """Use an LLM to semantically validate an extracted protocol graph.

    Args:
        graph: The extracted protocol graph JSON.
        section_text: The source text from the PDF section.
        model: OpenAI model to use.

    Returns:
        Validation result dict with issues and optional corrected graph.
    """
    client = OpenAI()

    # Strip internal metadata before sending to validator
    graph_clean = {k: v for k, v in graph.items()
                   if not k.startswith("_")}

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Review this protocol extraction.\n\n"
                f"--- SOURCE TEXT ---\n{section_text}\n\n"
                f"--- EXTRACTED GRAPH ---\n{json.dumps(graph_clean, indent=2)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    # Attach validation metadata
    result["_validation_meta"] = {
        "model": model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return result


# ── Combined validation pipeline ────────────────────────────────────

@dataclass
class ValidationResult:
    """Full validation result combining structural and semantic checks."""
    structural_issues: list[StructuralIssue]
    semantic_result: Optional[dict] = None
    original_graph: Optional[dict] = None
    corrected_graph: Optional[dict] = None
    was_corrected: bool = False
    total_issues: int = 0
    error_count: int = 0
    warning_count: int = 0
    validation_tokens: int = 0


def validate_protocol(
    graph: dict,
    section_text: str,
    run_semantic: bool = True,
    model: str = "gpt-4o",
) -> ValidationResult:
    """Run full validation pipeline on an extracted protocol graph.

    1. Structural checks (free)
    2. Semantic review (LLM, optional)
    3. Apply corrections if the LLM provides them

    Args:
        graph: Extracted protocol graph.
        section_text: Source text from the PDF section.
        run_semantic: Whether to run the LLM semantic validation.
        model: Model for semantic validation.

    Returns:
        ValidationResult with issues, corrections, and final graph.
    """
    result = ValidationResult(
        structural_issues=[],
        original_graph=graph,
    )

    # ── Step 1: Structural validation ────────────────────────────────
    result.structural_issues = validate_structure(graph)
    result.error_count = sum(1 for i in result.structural_issues if i.severity == "error")
    result.warning_count = sum(1 for i in result.structural_issues if i.severity == "warning")

    # ── Step 2: Semantic validation ──────────────────────────────────
    if run_semantic:
        semantic = validate_semantic(graph, section_text, model=model)
        result.semantic_result = semantic
        result.validation_tokens = semantic.get("_validation_meta", {}).get("total_tokens", 0)

        # Count semantic issues
        sem_issues = semantic.get("issues", [])
        result.error_count += sum(1 for i in sem_issues if i.get("severity") == "error")
        result.warning_count += sum(1 for i in sem_issues if i.get("severity") == "warning")

        # Apply corrections if provided
        corrected = semantic.get("corrected_graph")
        if corrected and semantic.get("corrections_made", False):
            # Re-run structural checks on the corrected graph
            corrected_structural = validate_structure(corrected)
            corrected_errors = sum(1 for i in corrected_structural if i.severity == "error")

            # Only accept correction if it doesn't introduce structural errors
            original_errors = sum(1 for i in result.structural_issues if i.severity == "error")
            if corrected_errors <= original_errors:
                result.corrected_graph = corrected
                result.was_corrected = True

    result.total_issues = result.error_count + result.warning_count
    return result


def print_validation_summary(result: ValidationResult) -> None:
    """Print a readable summary of validation results."""
    print(f"Structural: {len(result.structural_issues)} issues "
          f"({result.error_count} errors, {result.warning_count} warnings)")

    if result.structural_issues:
        for issue in result.structural_issues:
            marker = "ERROR" if issue.severity == "error" else "WARN"
            print(f"  [{marker}] {issue.message}")

    if result.semantic_result:
        sem = result.semantic_result
        passed = sem.get("review_passed", False)
        issues = sem.get("issues", [])
        print(f"Semantic: {'PASSED' if passed else 'ISSUES FOUND'} ({len(issues)} issues)")
        for issue in issues:
            sev = issue.get("severity", "?").upper()
            desc = issue.get("description", "?")
            itype = issue.get("type", "?")
            print(f"  [{sev}] ({itype}) {desc}")

        if result.was_corrected:
            summary = sem.get("correction_summary", "")
            print(f"Correction applied: {summary}")

    if result.validation_tokens:
        print(f"Validation tokens: {result.validation_tokens}")
