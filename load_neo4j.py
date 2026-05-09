"""
Load extracted protocols into Neo4j.

Usage:
    python load_neo4j.py [--uri bolt://localhost:7687] [--user neo4j] [--password password]
                         [--enrich] [--clear] [--stats]

Options:
    --uri       Neo4j connection URI (default: bolt://localhost:7687)
    --user      Neo4j username (default: neo4j)
    --password  Neo4j password (required)
    --enrich    Run KG enrichment before loading (adds roles, equipment, etc.)
    --clear     Clear the database before loading
    --stats     Print graph statistics after loading
    --dir       Directory containing extracted JSON files (default: evaluation/extracted)
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(description="Load protocol graphs into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--dir", default="evaluation/extracted")
    parser.add_argument("--enrich", action="store_true", help="Run KG enrichment first")
    parser.add_argument("--clear", action="store_true", help="Clear database before loading")
    parser.add_argument("--stats", action="store_true", help="Print stats after loading")

    args = parser.parse_args()

    # Optional enrichment pass
    if args.enrich:
        print("[neo4j] Running KG enrichment...")
        from src.graph.kg_enrichment import enrich_batch
        enrich_batch(args.dir, model="gpt-4o-mini")
        print()

    # Load into Neo4j
    from src.graph.neo4j_export import Neo4jExporter

    print(f"[neo4j] Connecting to {args.uri}...")
    with Neo4jExporter(args.uri, args.user, args.password, args.database) as exporter:
        if args.clear:
            print("[neo4j] Clearing database...")
            exporter.clear_database()

        print("[neo4j] Creating constraints...")
        exporter.create_constraints()

        print(f"[neo4j] Loading protocols from {args.dir}/...")
        results = exporter.load_all(args.dir)

        total_nodes = sum(r.get("nodes_created", 0) for r in results)
        total_edges = sum(r.get("edges_created", 0) for r in results)
        total_entities = sum(r.get("entities_created", 0) for r in results)
        errors = sum(1 for r in results if "error" in r)

        print(f"\n[neo4j] Done: {len(results) - errors} protocols loaded")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {total_edges}")
        print(f"  Total entities: {total_entities}")
        if errors:
            print(f"  Errors: {errors}")

        if args.stats:
            print("\n[neo4j] Graph statistics:")
            stats = exporter.get_graph_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
