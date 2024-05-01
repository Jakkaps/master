import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MP(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super(MP, self).__init__(aggr="mean")
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.self_lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        self_x = self.self_lin(x)
        return self.propagate(edge_index, x=x, self_x=self_x)

    def message(self, x_j):
        return self.lin(x_j)

    def update(self, aggr_out, self_x):
        return F.relu(aggr_out + self_x)
