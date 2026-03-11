# scripts/print_sel_edges.py
from pathlib import Path
import sys

proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from utils.graph_utils import build_edge_index

csv_path = proj_root / "data" / "ecity_edges.csv"
print("Reading edge file:", csv_path)

SEL, eid2idx, edge_index = build_edge_index(str(csv_path), N=30, device="cpu")
print("\nSEL_EDGES (first 30):")
print(SEL)
print("\nLength SEL_EDGES:", len(SEL))
