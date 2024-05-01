from statistics import mean

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import get_torch_device


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = torch.clamp(1 - target * output, min=0)
        return torch.mean(hinge_loss)


class ModelManager(nn.Module):
    def __init__(
        self, model, optimizer, criterion=HingeLoss(), device=get_torch_device()
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, loader, epochs, loss_window=10):
        self.model.train()

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
                progress_bar.set_postfix(
                    epoch=mean(batch_losses),
                    window=mean(batch_losses[-loss_window:]),
                )

            progress_bar.close()
            print(f"Epoch {epoch + 1}/{epochs} Loss: {mean(batch_losses)}\n")

    def eval(self, loader, loss_window=10):
        self.model.eval()

        batch_losses = []
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(loader), total=len(loader), desc="Evaluating", leave=True
            )
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch)
                loss = self.criterion(out, batch.y)

                batch_losses.append(loss.item())
                progress_bar.set_postfix(
                    avg=mean(batch_losses),
                    window=mean(batch_losses[-loss_window:]),
                )
                progress_bar.update(1)

        print(f"Eval loss: {mean(batch_losses)}")
