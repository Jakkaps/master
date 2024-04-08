import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops

from chat_dataset import ChatDataset


class RelationAwareMP(MessagePassing):
    def __init__(self, n_relations, in_channels, out_channels):
        super().__init__(aggr="add")
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
                edge_type=edge_type[mask],
                edge_weights=edge_weights,
            )

        return F.relu(out)

    def message(self, x_j, edge_index_i, edge_index_j, edge_type, edge_weights):
        type_idx = edge_type[0]

        norm = (
            edge_weights[edge_index_i, edge_index_j] / self.norm_constants[type_idx]
        ).unsqueeze(-1)

        out = norm * self.lins[type_idx](x_j)
        return out


class MP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MP, self).__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.self_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        self_x = self.self_lin(x)

        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, self_x=self_x
        )

    def message(self, x_j):
        return self.lin(x_j)

    def update(self, aggr_out, self_x):
        return F.relu(aggr_out + self_x)


class UtteranceEmbedding(nn.Module):
    """Embeds dialog utterances into a fixed-size vector."""

    def __init__(self):
        super(UtteranceEmbedding, self).__init__()
        self.st = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def forward(self, x):
        return self.st.encode(x, convert_to_tensor=True, batch_size=len(x))


class EvalNet(nn.Module):
    def __init__(
        self, n_layers=1, n_classes=5, n_relations=9, hidden_size=768, embed_size=384
    ):
        super(EvalNet, self).__init__()
        self.n_layers = n_layers

        self.embed = UtteranceEmbedding()

        self.relation_aware_mp = RelationAwareMP(
            n_relations=n_relations, in_channels=embed_size, out_channels=hidden_size
        )

        self.mp = MP(in_channels=hidden_size, out_channels=hidden_size)

        self.lin = nn.Linear(hidden_size, n_classes)

    def forward(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_attr
        x = (
            [utt for sublist in batch.x for utt in sublist]
            if len(batch.x) > 1
            else batch.x[0]
        )

        x = self.embed(x)

        # Construct edge weights
        edge_weights = x @ x.T

        # Process the dialog graph
        for _ in range(self.n_layers):
            x = self.relation_aware_mp(x, edge_index, edge_weights, edge_type)
            x = self.mp(x, edge_index)

        # Aggregate node embeddings to dialog embedding
        x = x.mean(dim=0)

        # Compute the final score from dialog embedding
        return self.lin(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    eval_net = EvalNet()
    eval_net.to(device)

    chat_dataset = ChatDataset(root="data", dataset="gogi_chats")
    loader = DataLoader(chat_dataset, batch_size=1)

    for batch in loader:
        batch = batch.to(device)
        out = eval_net(batch)
        print(out)
        break
