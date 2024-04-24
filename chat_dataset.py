import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


class ChatDataset(InMemoryDataset):
    def __init__(
        self, root, dataset, transform=None, pre_transform=None, pre_filter=None
    ):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.dataset}_processed.pt"]

    def process(self):
        nodes = torch.load(f"{self.root}/{self.dataset}_nodes.pt")
        edge_idxs = torch.load(f"{self.root}/{self.dataset}_edge_idxs.pt")
        edges = torch.load(f"{self.root}/{self.dataset}_edges.pt")
        labels = torch.load(f"{self.root}/{self.dataset}_labels.pt")

        data_list = [
            Data(
                x1=nodes[i][0],
                edge_index1=edge_idxs[i][0],
                edge_attr1=edges[i][0],
                x2=nodes[i][1],
                edge_index2=edge_idxs[i][1],
                edge_attr2=edges[i][1],
                y=labels[i],
                num_nodes=len(nodes[i][1]),
            )
            for i in range(len(nodes))
        ]

        self.save(data_list, self.processed_paths[0])
