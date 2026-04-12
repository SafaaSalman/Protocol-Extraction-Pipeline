"""
Side-by-side visualization of protocol graphs:
  Gold standard  |  Text-only extraction  |  Multimodal extraction

Produces a single PNG image for easy comparison.
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# --- Style constants ---
NODE_COLORS = {
    "start": "#4CAF50",     # green
    "end": "#F44336",       # red
    "action": "#2196F3",    # blue
    "decision": "#FF9800",  # orange
}
NODE_SHAPES = {
    "start": "o",       # circle
    "end": "s",         # square
    "action": "o",      # circle
    "decision": "D",    # diamond
}
EDGE_COLOR = "#555555"
EDGE_LABEL_COLOR = "#333333"


def wrap_label(text: str, width: int = 18) -> str:
    """Wrap long node labels for display."""
    text = text.strip()
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width))


def build_nx_graph(protocol: dict) -> nx.DiGraph:
    """Convert a protocol JSON dict to a NetworkX DiGraph."""
    G = nx.DiGraph()
    for node in protocol.get("nodes", []):
        G.add_node(
            node["id"],
            label=node["text"],
            node_type=node["type"],
        )
    for edge in protocol.get("edges", []):
        G.add_edge(
            edge["from"],
            edge["to"],
            condition=edge.get("condition", ""),
        )
    return G


def hierarchical_layout(G: nx.DiGraph) -> dict:
    """Compute a top-down hierarchical layout for a DAG-like graph.

    Uses topological generations for Y, spreads nodes evenly in X per layer.
    Falls back to spring layout for cyclic graphs.
    """
    try:
        # Find start node (or first node) as root
        start_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "start"]
        if not start_nodes:
            start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not start_nodes:
            start_nodes = [list(G.nodes())[0]]

        # BFS-based layering from start
        layers = {}
        visited = set()
        queue = [(start_nodes[0], 0)]
        visited.add(start_nodes[0])

        while queue:
            node, depth = queue.pop(0)
            layers[node] = depth
            for succ in G.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, depth + 1))

        # Handle disconnected nodes
        for n in G.nodes():
            if n not in layers:
                layers[n] = max(layers.values(), default=0) + 1

        # Group by layer
        layer_groups = {}
        for node, layer in layers.items():
            layer_groups.setdefault(layer, []).append(node)

        # Sort nodes within each layer for consistent ordering
        for layer in layer_groups:
            layer_groups[layer].sort()

        # Compute positions
        max_layer = max(layer_groups.keys()) if layer_groups else 0
        pos = {}
        for layer, nodes in layer_groups.items():
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i - (n_nodes - 1) / 2) * 1.8
                y = -layer * 1.5
                pos[node] = (x, y)

        return pos
    except Exception:
        return nx.spring_layout(G, k=2, iterations=50, seed=42)


def draw_protocol_graph(
    ax: plt.Axes,
    protocol: dict,
    title: str,
    highlight_matched: set | None = None,
):
    """Draw a protocol graph on a matplotlib Axes."""
    G = build_nx_graph(protocol)

    if len(G.nodes()) == 0:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    pos = hierarchical_layout(G)

    # Draw edges first (behind nodes)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        cond = data.get("condition", "")
        if cond:
            edge_labels[(u, v)] = cond

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        width=1.2,
        min_source_margin=18,
        min_target_margin=18,
    )

    # Edge labels
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=6,
        font_color=EDGE_LABEL_COLOR,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        rotate=False,
    )

    # Draw nodes by type for distinct styling
    for node_type, color in NODE_COLORS.items():
        nodelist = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        if not nodelist:
            continue

        node_size = 700 if node_type == "decision" else 500

        # Dimming for unmatched nodes (optional highlighting)
        alphas = []
        edge_colors = []
        for n in nodelist:
            if highlight_matched is not None and n not in highlight_matched:
                alphas.append(0.35)
                edge_colors.append("#999999")
            else:
                alphas.append(1.0)
                edge_colors.append("#333333")

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=nodelist,
            node_color=color,
            node_shape=NODE_SHAPES.get(node_type, "o"),
            node_size=node_size,
            alpha=alphas,
            edgecolors=edge_colors,
            linewidths=1.5,
        )

    # Node labels
    labels = {}
    for n, d in G.nodes(data=True):
        labels[n] = f"{n}\n{wrap_label(d.get('label', n), 16)}"

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=labels,
        font_size=5.5,
        font_weight="bold",
    )

    # Title and stats
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    types = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "?")
        types[t] = types.get(t, 0) + 1
    type_str = " ".join(f"{v}{k[0].upper()}" for k, v in sorted(types.items()))

    ax.set_title(
        f"{title}\n{n_nodes} nodes, {n_edges} edges  [{type_str}]",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.axis("off")


def visualize_comparison(
    gold_path: str,
    text_path: str,
    multimodal_path: str,
    output_path: str = "evaluation/comparison_triage.png",
):
    """Create a 3-panel side-by-side comparison figure."""
    with open(gold_path) as f:
        gold = json.load(f)
    with open(text_path) as f:
        text_only = json.load(f)
    with open(multimodal_path) as f:
        multimodal = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(24, 14))
    fig.suptitle(
        "Protocol Graph Comparison: Multi-Casualty Triage System",
        fontsize=16, fontweight="bold", y=0.97,
    )

    draw_protocol_graph(axes[0], gold, "Gold Standard")
    draw_protocol_graph(axes[1], text_only, "Text-Only Extraction")
    draw_protocol_graph(axes[2], multimodal, "Multimodal Extraction\n(Text + Page Image)")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=NODE_COLORS["start"], label="Start", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["end"], label="End", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["action"], label="Action", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["decision"], label="Decision", edgecolor="#333"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved comparison to: {output_path}")
    plt.close(fig)


def visualize_all_protocols(
    output_path: str = "evaluation/comparison_all.png",
):
    """Create a full comparison grid: all 3 protocols x (gold vs extracted)."""
    protocols = [
        {
            "name": "Risk Management Process",
            "gold": "evaluation/gold/risk_management_01.json",
            "extracted": "evaluation/extracted/risk_management_process.json",
        },
        {
            "name": "Patient Assessment",
            "gold": "evaluation/gold/patient_assessment_01.json",
            "extracted": "evaluation/extracted/patient_assessment.json",
        },
        {
            "name": "Multi-Casualty Triage",
            "gold": "evaluation/gold/triage_01.json",
            "extracted": "evaluation/extracted/multi_casualty_triage_system.json",
            "multimodal": "evaluation/extracted/triage_multimodal.json",
        },
    ]

    # 3 rows (protocols) x 3 cols (gold / text-only / multimodal where available)
    n_rows = len(protocols)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, n_rows * 11))
    fig.suptitle(
        "Protocol Extraction Comparison: Gold vs Text-Only vs Multimodal",
        fontsize=18, fontweight="bold", y=0.98,
    )

    for row, prot in enumerate(protocols):
        with open(prot["gold"]) as f:
            gold = json.load(f)
        with open(prot["extracted"]) as f:
            text_ext = json.load(f)

        draw_protocol_graph(axes[row][0], gold, f"{prot['name']}\nGold Standard")
        draw_protocol_graph(axes[row][1], text_ext, f"{prot['name']}\nText-Only")

        if "multimodal" in prot and Path(prot["multimodal"]).exists():
            with open(prot["multimodal"]) as f:
                mm = json.load(f)
            draw_protocol_graph(axes[row][2], mm, f"{prot['name']}\nMultimodal")
        else:
            axes[row][2].text(
                0.5, 0.5, "N/A\n(text-only protocol)",
                ha="center", va="center", fontsize=14, color="#999",
                transform=axes[row][2].transAxes,
            )
            axes[row][2].set_title(f"{prot['name']}\nMultimodal", fontsize=11, fontweight="bold", pad=12)
            axes[row][2].axis("off")

    legend_elements = [
        mpatches.Patch(facecolor=NODE_COLORS["start"], label="Start", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["end"], label="End", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["action"], label="Action", edgecolor="#333"),
        mpatches.Patch(facecolor=NODE_COLORS["decision"], label="Decision", edgecolor="#333"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved full comparison to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Triage-focused comparison (3 panels)
    visualize_comparison(
        gold_path="evaluation/gold/triage_01.json",
        text_path="evaluation/extracted/multi_casualty_triage_system.json",
        multimodal_path="evaluation/extracted/triage_multimodal.json",
        output_path="evaluation/comparison_triage.png",
    )

    # Full comparison grid (all protocols)
    visualize_all_protocols(
        output_path="evaluation/comparison_all.png",
    )
