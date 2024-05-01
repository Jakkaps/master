import torch
import torch.nn as nn

from model.graph_embedding import GraphEmbedding


class DialogRater(nn.Module):
    def __init__(
        self,
        n_graph_layers=1,
        n_graph_relations=9,
        embed_dim=384,
        graph_hidden_size=384,
        graph_out_size=10,
        n_dimensions=5,
    ):
        super(DialogRater, self).__init__()

        self.graph_embed = GraphEmbedding(
            n_layers=n_graph_layers,
            n_relations=n_graph_relations,
            embed_dim=embed_dim,
            hidden_dim=graph_hidden_size,
            out_dim=graph_out_size,
        )
        self.lin = nn.Linear(graph_out_size, n_dimensions)

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_attr
        batch_size = batch.num_graphs

        # Compute dialog embeddings
        x = self.graph_embed(x, edge_index, edge_type, batch_size)

        return self.lin(x).squeeze()
