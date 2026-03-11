import torch
from torch import nn
from torch_geometric.nn import GATConv


class HybridGAT_LSTM(nn.Module):
    """
    EXACT architecture used during training:
    - gat_heads = 4
    - concat = False
    - gat_hidden = 64
    - lstm_hidden = 128
    - fusion input = 192 (128 + 64)
    """

    def __init__(self, in_dim, gat_hidden=64, gat_heads=4,
                 lstm_hidden=128, fusion_hidden=128):
        super().__init__()

        # GAT layers (concat=False → output dim = gat_hidden = 64)
        self.gat1 = GATConv(
            in_dim,
            gat_hidden,
            heads=gat_heads,
            concat=False
        )

        self.gat2 = GATConv(
            gat_hidden,
            gat_hidden,
            heads=gat_heads,
            concat=False
        )

        # LSTM input = 64
        self.lstm = nn.LSTM(
            input_size=gat_hidden,        # 64
            hidden_size=lstm_hidden,      # 128
            batch_first=True
        )

        # Fusion → (128 LSTM + 64 GAT = 192)
        self.fusion = nn.Linear(lstm_hidden + gat_hidden, fusion_hidden)

        # Output per-edge prediction
        self.out = nn.Linear(fusion_hidden, 1)

    def forward(self, x_seq, edge_index):
        B, T, N, F = x_seq.shape
        outputs = []

        for b in range(B):
            spatial_seq = []

            for t in range(T):
                xt = x_seq[b, t]                 # (N, F)

                g = torch.relu(self.gat1(xt, edge_index))   # (N, 64)
                g = torch.relu(self.gat2(g, edge_index))    # (N, 64)

                spatial_seq.append(g.unsqueeze(0))          # (1, N, 64)

            spatial_seq = torch.cat(spatial_seq, dim=0)     # (T, N, 64)
            spatial_seq = spatial_seq.permute(1, 0, 2)       # (N, T, 64)

            lstm_out, _ = self.lstm(spatial_seq)            # (N, T, 128)
            lstm_last = lstm_out[:, -1, :]                  # (N, 128)

            gat_last = spatial_seq[:, -1, :]                # (N, 64)

            fused = torch.cat([lstm_last, gat_last], dim=1) # (N, 192)
            fused = torch.relu(self.fusion(fused))          # (N, 128)

            y = self.out(fused).squeeze(-1)                 # (N,)

            outputs.append(y.unsqueeze(0))

        return torch.cat(outputs, dim=0)                    # (B, N)
