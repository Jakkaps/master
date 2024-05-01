import torch
import torch.nn as nn

from model.graph_embedding import GraphEmbedding


class DialogDiscriminator(nn.Module):
    def __init__(
        self,
        n_graph_layers=1,
        n_graph_relations=9,
        embed_dim=384,
        graph_hidden_dim=384,
        graph_out_dim=10,
    ):
        super(DialogDiscriminator, self).__init__()

        self.graph_embed = GraphEmbedding(
            n_layers=n_graph_layers,
            n_relations=n_graph_relations,
            embed_dim=embed_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
        )
        self.lin = nn.Linear(2 * graph_out_dim, 1)

    def forward(self, batch):
        x1, edge_index1, edge_type1 = batch.x1, batch.edge_index1, batch.edge_attr1
        x2, edge_index2, edge_type2 = batch.x2, batch.edge_index2, batch.edge_attr2

        batch_size = batch.num_graphs

        # Compute dialog embeddings
        x1 = self.graph_embed(x1, edge_index1, edge_type1, batch_size)
        x2 = self.graph_embed(x2, edge_index2, edge_type2, batch_size)

        # Concatenate dialog embeddings for both graphs
        x = torch.cat([x1, x2], dim=1)

        # Compute the final score from dialog embeddings
        return self.lin(x).squeeze()
