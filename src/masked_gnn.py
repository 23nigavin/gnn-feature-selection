import torch
import torch.nn as nn
from gnn import GCN


class MaskedGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, k=None):
        super().__init__()

        self.feature_logits = nn.Parameter(0.01 * torch.randn(num_features))
        self.k = k

        self.gcn = GCN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def get_mask(self):
        soft_mask = torch.sigmoid(self.feature_logits)

        if self.k is None:
            return soft_mask

        _, top_indices = torch.topk(soft_mask, self.k)

        hard_mask = torch.zeros_like(soft_mask)
        hard_mask[top_indices] = 1.0

        return hard_mask + soft_mask - soft_mask.detach()

    def forward(self, x, edge_index):
        mask = self.get_mask()
        x_masked = x * mask
        return self.gcn(x_masked, edge_index)

    def mask_l1_penalty(self):
        return torch.sigmoid(self.feature_logits).mean()