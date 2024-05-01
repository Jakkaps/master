import sys

import click
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dialog_discrimination_dataset import DialogDiscriminationDataset
from model.dialog_discriminator import DialogDiscriminator
from model_manager import ModelManager
from utils import get_base_filename, get_torch_device


@click.command()
@click.option("--mode", default="train", help="Mode of operation")
@click.option("--lr", default=0.001, help="Learning rate")
@click.option("--epochs", default=1, help="Number of epochs")
@click.option("--batch_size", default=16, help="Batch size")
@click.option("--n_training_points", type=int, help="Number of training points")
@click.option("--n_layers", default=1, type=int, help="Number of layers")
@click.option("--graph_out_dim", default=10, type=int, help="Graph output dimension")
def main(
    mode: str,
    lr: float,
    epochs: int,
    batch_size: int,
    n_training_points: int,
    n_layers: int,
    graph_out_dim: int,
):
    name_base = get_base_filename(
        lr, epochs, batch_size, n_training_points, n_layers, graph_out_dim
    )
    log_file = open(
        f"logs/{mode}/{name_base}.log",
        "w+",
    )
    sys.stdout = log_file

    device = get_torch_device()

    model = DialogDiscriminator(n_graph_layers=n_layers, graph_out_dim=graph_out_dim)
    model.to(device)

    root = "data"
    dataset = "twitter_cs"
    model_path = f"ckpts/{name_base}.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    manager = ModelManager(model, optimizer)

    if mode == "train":
        data = DialogDiscriminationDataset(root=root, dataset=dataset, split="train")
        data = Subset(data, range(n_training_points)) if n_training_points else data
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        manager.train(epochs=epochs, loader=loader)
        manager.save(model_path)
    elif mode == "eval":
        data = DialogDiscriminationDataset(root=root, dataset=dataset, split="test")
        data = Subset(data, range(n_training_points)) if n_training_points else data
        loader = DataLoader(data, batch_size=batch_size)

        manager.load(model_path)
        manager.eval(loader)

    log_file.close()


if __name__ == "__main__":
    main()
