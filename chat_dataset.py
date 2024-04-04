import torch
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
        node_encodings = torch.load(f"{self.root}/{self.dataset}_node_encodings.pt")
        edge_idxs = torch.load(f"{self.root}/{self.dataset}_edge_idxs.pt")
        edges = torch.load(f"{self.root}/{self.dataset}_edges.pt")
        labels = torch.load(f"{self.root}/{self.dataset}_labels.pt")

        data_list = [
            Data(
                x=node_encoding,
                edge_index=edge_idx,
                edge_attr=edge,
                y=label,
            )
            for node_encoding, edge_idx, edge, label in zip(
                node_encodings, edge_idxs, edges, labels
            )
        ]

        self.save(data_list, self.processed_paths[0])
