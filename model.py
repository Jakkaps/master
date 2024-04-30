import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from transformers import AutoModel

from data import ChatDataset


def print_tensor_stats(tensor):
    row_means = tensor.mean(dim=1)
    row_stds = tensor.std(dim=1)
    print(f"Mean: {row_means.mean():.2f},\tStd: {row_stds.mean():.2f}")


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


class MP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MP, self).__init__(aggr="mean")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.self_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        self_x = self.self_lin(x)
        return self.propagate(edge_index, x=x, self_x=self_x)

    def message(self, x_j):
        return self.lin(x_j)

    def update(self, aggr_out, self_x):
        return F.relu(aggr_out + self_x)


class UtteranceEmbedding(nn.Module):
    """Embeds dialog utterances into a fixed-size vector."""

    def __init__(self, embed_size):
        super(UtteranceEmbedding, self).__init__()

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
        )
        model = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        self.model = get_peft_model(model, peft_config)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, x):
        attn_mask = x.ne(0).int()
        out = self.model(x, attention_mask=attn_mask)
        embeddings = out.last_hidden_state[
            :, 0, :
        ]  # Index 0 for the [CLS] token in each sequence
        return self.bn(embeddings)


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


class DialogDiscriminator(nn.Module):
    def __init__(
        self,
        n_graph_layers=1,
        n_graph_relations=9,
        hidden_size=384,
        embed_size=384,
    ):
        super(DialogDiscriminator, self).__init__()

        self.graph_embed = GraphEmbedding(
            n_layers=n_graph_layers,
            n_relations=n_graph_relations,
            hidden_size=hidden_size,
            embed_size=embed_size,
        )
        self.lin = nn.Linear(2 * hidden_size, 1)
        self.tanh = nn.Tanh()

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
        return self.tanh(self.lin(x).squeeze())
