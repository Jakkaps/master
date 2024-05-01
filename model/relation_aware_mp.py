import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class RelationAwareMP(MessagePassing):
    def __init__(self, n_relations, in_channels, out_channels):
        super().__init__(aggr="mean")
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.lins = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(n_relations)]
        )
        self.norm_constants = nn.Parameter(torch.ones(n_relations))

    def forward(self, x, edge_index, edge_weights, edge_type):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for relation_type in range(self.n_relations):
            mask = edge_type == relation_type
            relation_idxs = edge_index[:, mask]

            if relation_idxs.size(1) == 0:
                continue

            out += self.propagate(
                relation_idxs,
                x=x,
                edge_weights=edge_weights,
                type_idx=relation_type,
            )

        return F.relu(out)

    def message(self, x_j, edge_index_i, edge_index_j, edge_weights, type_idx):
        norm = (
            edge_weights[edge_index_i, edge_index_j] / self.norm_constants[type_idx]
        ).unsqueeze(-1)

        out = norm * self.lins[type_idx](x_j)
        return out
