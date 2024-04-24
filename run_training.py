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

        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.update(1)

                if (i + 1) % 100 == 0 or i == len(loader) - 1:
                    progress_bar.set_postfix(loss=running_loss / ((i % 100) + 1))
                    running_loss = 0.0

    def eval(self, loader):
        self.model.eval()

        total_loss = 0
        num_graphs = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                outputs = model(data)
                targets = data.y
                loss = criterion(outputs, targets.view(-1))
                total_loss += loss.item()
                num_graphs += 1

        average_loss = total_loss / num_graphs
        print(f"Average Loss: {average_loss}")


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.cuda.empty_cache()  # Clear memory cache on the GPU if available

    config = {
        "lr": 0.001,
        "epochs": 1,
        "batch_size": 16,
    }

    model = DialogDiscriminator()
    model.to(device)

    train_dataset = ChatDataset(root="data", dataset="twitter_cs", split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    trainer = EvalNetTrainer(model, optimizer, criterion, device)
    trainer.train(epochs=config["epochs"], loader=train_loader)

    test_dataset = ChatDataset(root="data", dataset="twitter_cs", split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )
    trainer.eval(loader=train_loader)

    trainer.save("models/trained_model.pth")
