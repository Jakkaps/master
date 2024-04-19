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
        nodes = torch.load(f"{self.root}/{self.dataset}_nodes.pt")
        edge_idxs = torch.load(f"{self.root}/{self.dataset}_edge_idxs.pt")
        edges = torch.load(f"{self.root}/{self.dataset}_edges.pt")
        labels = torch.load(f"{self.root}/{self.dataset}_labels.pt")

        def construct_data(idx, graph_num):
            return Data(
                x=nodes[idx][graph_num],
                edge_index=edge_idxs[idx][graph_num],
                edge_attr=edges[idx][graph_num],
                num_nodes=len(nodes[idx][graph_num]),
            )

        data_list = [
            Data(
                x1=construct_data(i, 0),
                x2=construct_data(i, 1),
                y=labels[i],
            )
            for i in range(len(nodes))
        ]

        self.save(data_list, self.processed_paths[0])
