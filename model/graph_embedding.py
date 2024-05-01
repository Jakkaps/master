import torch.nn as nn

from model.mp import MP
from model.relation_aware_mp import RelationAwareMP
from model.utterance_embedding import UtteranceEmbedding


def pairwise_cosine_similarity(x):
    x_norm = x / x.norm(dim=1, keepdim=True)
    return x_norm @ x_norm.T


class GraphEmbedding(nn.Module):
    def __init__(self, n_layers, n_relations, hidden_size, embed_size):
        super(GraphEmbedding, self).__init__()

        self.n_layers = n_layers
        self.embed = UtteranceEmbedding(embed_size=embed_size)

        relation_aware_mps = []
        mps = []

        current_size = embed_size
        for _ in range(n_layers):
            relation_aware_mps.append(
                RelationAwareMP(
                    n_relations=n_relations,
                    in_channels=current_size,
                    out_channels=hidden_size,
                )
            )
            mps.append(MP(in_channels=hidden_size, out_channels=hidden_size))
            current_size = hidden_size

        self.relation_aware_mps = nn.ModuleList(relation_aware_mps)
        self.mps = nn.ModuleList(mps)
        self.lin = nn.Linear(hidden_size, 1)

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

        return x
