# scripts/list_edges.py
import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).resolve().parents[1]   # project root
    csv_path = base / "data" / "ecity_edges.csv"

    if not csv_path.exists():
        print("ERROR: expected file not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    print("File loaded:", csv_path)
    print("Columns:", df.columns.tolist())
    print("\n-- First 10 rows --")
    print(df.head(10).to_string(index=False))

    # Determine edge id column
    if 'edge_id' in df.columns:
        id_col = 'edge_id'
    elif 'id' in df.columns:
        id_col = 'id'
    else:
        # fallback to first column name
        id_col = df.columns[0]
        print(f"\nNote: using first column as edge id -> '{id_col}'")

    ids = df[id_col].tolist()
    print(f"\nTotal edges: {len(ids)}")
    print("Sample edge ids (first 30):", ids[:30])

    # show adjacency candidates if u/v present
    if {'u', 'v'}.issubset(df.columns):
        print("\nFound 'u' and 'v' columns. Sample (edge_id, u, v):")
        print(df[[id_col, 'u', 'v']].head(20).to_string(index=False))
        # display adjacency counts per node
        node_counts = {}
        for r in df[[id_col,'u','v']].itertuples(index=False):
            _, u, v = r
            node_counts[u] = node_counts.get(u, 0) + 1
            node_counts[v] = node_counts.get(v, 0) + 1
        top_nodes = sorted(node_counts.items(), key=lambda x: -x[1])[:10]
        print("\nTop nodes by connected edges (node, count):", top_nodes)
    else:
        print("\nNo 'u'/'v' columns detected. If your CSV uses geometry, inspect geometry column above.")

    print("\n--- How to pick demo pairs ---")
    print("1) Adjacent: pick two edges whose u/v share a node (see the u/v table above).")
    print("2) Same edge: pick any single id (start=end).")
    print("3) Moderate distance: pick ids spaced roughly in the middle of the list.")
    print("4) Far: pick first and last ids printed above.")
    print("\nIf you want, paste the first 15 rows of the output here and I will choose 4 demo pairs for you.")

if __name__ == "__main__":
    main()
