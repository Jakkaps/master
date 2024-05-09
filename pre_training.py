import sys

import click
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from dialog_discrimination_dataset import DialogDiscriminationDataset
from model.dialog_discriminator import DialogDiscriminator
from model_manager import ModelManager
from utils import get_file_names, get_torch_device


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
    log_name, model_name = get_file_names(
        lr, epochs, batch_size, n_training_points, n_layers, graph_out_dim
    )
    log_file = open(
        f"logs/{mode}/{log_name}",
        "w+",
    )
    # sys.stdout = log_file

    device = get_torch_device()

    model = DialogDiscriminator(n_graph_layers=n_layers, graph_out_dim=graph_out_dim)
    model.to(device)

    for param in model.graph_embed.embed.model.parameters():
        param.requires_grad = False

    dataset = "twitter_cs"
    root = f"data/{dataset}"
    model_path = f"ckpts/{model_name}"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    manager = ModelManager(model, optimizer, "ckpts/" + model_name)

    train_data = DialogDiscriminationDataset(root=root, dataset=dataset, split="train")
    train_data = (
        Subset(train_data, range(n_training_points))
        if n_training_points
        else train_data
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = DialogDiscriminationDataset(root=root, dataset=dataset, split="test")
    test_data = (
        Subset(test_data, range(n_training_points)) if n_training_points else test_data
    )
    eval_loader = DataLoader(test_data, batch_size=batch_size)

    if mode == "train":
        manager.train(
            epochs=epochs,
            train_loader=train_loader,
            eval_loader=eval_loader,
            save_every_epoch=True,
        )
        manager.save(model_path)
    elif mode == "eval":
        manager.load(model_path)
        manager.eval(eval_loader)

    log_file.close()


if __name__ == "__main__":
    main()
