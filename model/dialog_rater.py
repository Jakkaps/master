import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph_embedding import GraphEmbedding


def print_tensor_stats(tensor):
    row_means = tensor.mean(dim=1)
    row_stds = tensor.std(dim=1)
    print(f"Mean: {row_means.mean():.2f},\tStd: {row_stds.mean():.2f}")


class DialogRater(nn.Module):
    def __init__(
        self,
        n_graph_layers=1,
        n_graph_relations=9,
        embed_dim=384,
        graph_hidden_size=384,
        graph_out_dim=10,
        n_dimensions=4,
        n_hidden_layers=1,
        hidden_dim=50,
    ):
        super(DialogRater, self).__init__()

        self.graph_embed = GraphEmbedding(
            n_layers=n_graph_layers,
            n_relations=n_graph_relations,
            embed_dim=embed_dim,
            hidden_dim=graph_hidden_size,
            out_dim=graph_out_dim,
        )
        self.bn = nn.BatchNorm1d(graph_out_dim)

        layers = []
        activations = []
        in_dim = graph_out_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            activations.append(nn.ReLU())
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        self.out_lin = nn.Linear(
            hidden_dim if n_hidden_layers > 0 else graph_out_dim, n_dimensions
        )

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_attr
        batch_size = batch.num_graphs

        # Compute dialog embeddings
        x = self.graph_embed(x, edge_index, edge_type, batch_size)
        x = self.bn(x)

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = activation(layer(x))

        return self.out_lin(x)
