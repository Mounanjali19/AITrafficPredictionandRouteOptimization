import pandas as pd
import torch
from collections import defaultdict


def build_edge_index(edge_csv_path, N=30, device="cpu"):
    """
    Build edge_index in 'edge space' using your ecity_edges.csv structure.

    edge_csv_path: path to CSV with columns ['edge_id', 'u', 'v']
    N: number of edges to keep (first N unique edge_ids)
    device: 'cpu' or 'cuda'
    """
    edges_df = pd.read_csv(edge_csv_path)[["edge_id", "u", "v"]]

    # Take first N unique edges
    ALL = sorted(edges_df.edge_id.unique().tolist())
    SEL = ALL[:N]

    # Map edge_id -> 0..N-1
    eid2idx = {eid: i for i, eid in enumerate(SEL)}

    node2edges = defaultdict(list)

    for r in edges_df.itertuples(index=False):
        if r.edge_id in eid2idx:
            node2edges[r.u].append(r.edge_id)
            node2edges[r.v].append(r.edge_id)

    pairs = set()
    for lst in node2edges.values():
        # Keep only edges in SEL
        s = sorted(set([e for e in lst if e in eid2idx]))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pairs.add((eid2idx[s[i]], eid2idx[s[j]]))
                pairs.add((eid2idx[s[j]], eid2idx[s[i]]))

    if len(pairs) == 0:
        raise RuntimeError("No adjacent edges found when building edge_index.")

    src, dst = zip(*pairs)
    edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long).to(device)

    return SEL, eid2idx, edge_index
