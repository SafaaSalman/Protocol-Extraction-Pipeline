"""Test PET dataset adapter: download, convert, and save sample gold files."""
import sys
sys.path.insert(0, ".")
from src.graph.pet_adapter import PETAdapter

adapter = PETAdapter()
adapter.download()
protocols = adapter.convert_all()

# Show summary stats
print(f"\nTotal protocols converted: {len(protocols)}")
for p in protocols[:5]:
    nodes = len(p["nodes"])
    edges = len(p["edges"])
    decisions = sum(1 for n in p["nodes"] if n["type"] == "decision")
    title = p["title"][:50]
    print(f"  {title:50s} nodes={nodes} edges={edges} decisions={decisions}")
print("...")

# Save a sample
adapter.save_as_gold("evaluation/pet_gold/", protocols[:5])
print("\nSample gold files saved to evaluation/pet_gold/")
