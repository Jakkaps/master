import torch
import torch.nn as nn

from model.graph_embedding import GraphEmbedding


class DialogRater(nn.Module):
    def __init__(
        self,
        n_graph_layers=1,
        n_graph_relations=9,
        hidden_size=384,
        embed_size=384,
        n_dimensions=5,
    ):
        super(DialogRater, self).__init__()

        self.graph_embed = GraphEmbedding(
            n_layers=n_graph_layers,
            n_relations=n_graph_relations,
            hidden_size=hidden_size,
            embed_size=embed_size,
        )
        self.lin = nn.Linear(hidden_size, n_dimensions)

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_attr
        batch_size = batch.num_graphs

        # Compute dialog embeddings
        x = self.graph_embed(x, edge_index, edge_type, batch_size)

        return self.lin(x).squeeze()
