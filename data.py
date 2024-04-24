import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


class ChatDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        split="train",
    ):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if split == "train" else self.processed_paths[1]
        self.load(path)

    @property
    def processed_file_names(self):
        return [f"{self.dataset}_train.pt", f"{self.dataset}_test.pt"]

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

        random.shuffle(data_list)
        split_idx = int(0.8 * len(data_list))

        train_data = data_list[:split_idx]
        test_data = data_list[split_idx:]

        self.save(train_data, self.processed_paths[0])
        self.save(test_data, self.processed_paths[1])
