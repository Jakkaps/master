import torch.nn as nn

from model.mp import MP
from model.relation_aware_mp import RelationAwareMP
from model.utterance_embedding import UtteranceEmbedding


def pairwise_cosine_similarity(x):
    x_norm = x / x.norm(dim=1, keepdim=True)
    return x_norm @ x_norm.T


class GraphEmbedding(nn.Module):
    def __init__(self, n_layers, n_relations, embed_dim, hidden_dim, out_dim):
        super(GraphEmbedding, self).__init__()

        self.n_layers = n_layers
        self.embed = UtteranceEmbedding(embed_dim=embed_dim)

        relation_aware_mps = []
        mps = []

        for i in range(n_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            relation_aware_mps.append(
                RelationAwareMP(
                    n_relations=n_relations,
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                )
            )
            mps.append(MP(in_dim=hidden_dim, out_dim=hidden_dim))

        self.relation_aware_mps = nn.ModuleList(relation_aware_mps)
        self.mps = nn.ModuleList(mps)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type, batch_size):
        # Embed utterances
        x = self.embed(x)

        # Construct edge weights
        edge_weights = pairwise_cosine_similarity(x)

        # Process the dialog graph
        for i in range(self.n_layers):
            x = (
                self.mps[i](
                    self.relation_aware_mps[i](x, edge_index, edge_weights, edge_type),
                    edge_index,
                )
            ) + x

        # Aggregate to graph level
        x = x.view(batch_size, -1, x.size(-1))
        x = x.mean(dim=1)

        return self.lin(x)
