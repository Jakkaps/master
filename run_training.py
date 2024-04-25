import sys
from statistics import mean

import click
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import ChatDataset
from model import DialogDiscriminator


class EvalNetTrainer(nn.Module):
    def __init__(self, model, optimizer, criterion, device):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, loader, epochs):
        self.model.train()

        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = []
            progress_bar = tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

                progress_bar.update(1)
                progress_bar.set_postfix(loss=mean(batch_losses))

            progress_bar.close()
            epoch_losses.append(mean(batch_losses))
            print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_losses[-1]}\n")

        return epoch_losses

    def eval(self, loader):
        self.model.eval()

        total_loss = 0
        num_graphs = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = self.model(data)
                targets = data.y
                loss = self.criterion(outputs, targets.view(-1))
                total_loss += loss.item()
                num_graphs += 1

        average_loss = total_loss / num_graphs
        print(f"Eval loss: {average_loss}")


@click.command()
@click.option("--lr", default=0.001, help="Learning rate")
@click.option("--epochs", default=1, help="Number of epochs")
@click.option("--batch_size", default=16, help="Batch size")
@click.option("--n_training_points", type=int, help="Number of training points")
def main(
    lr: float,
    epochs: int,
    batch_size: int,
    n_training_points: int,
):
    log_file = open(f"logs/lr={lr}_epochs={epochs}_batch_size={batch_size}.log", "w+")
    sys.stdout = log_file

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.cuda.empty_cache()  # Clear memory cache on the GPU if available

    model = DialogDiscriminator()
    model.to(device)

    train_dataset = ChatDataset(root="data", dataset="twitter_cs", split="train")
    train_dataset = (
        Subset(train_dataset, range(n_training_points))
        if n_training_points
        else train_dataset
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = EvalNetTrainer(model, optimizer, criterion, device)
    epoch_losses = trainer.train(epochs=epochs, loader=train_loader)

    test_dataset = ChatDataset(root="data", dataset="twitter_cs", split="test")
    test_dataset = (
        Subset(test_dataset, range(n_training_points))
        if n_training_points
        else test_dataset
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.eval(loader=test_loader)

    model_name = f"models/lr={lr}_epochs={epochs}_batch_size={batch_size}_model.pth"
    trainer.save(model_name)

    log_file.close()


if __name__ == "__main__":
    main()
