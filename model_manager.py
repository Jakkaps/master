from statistics import mean

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm

from utils import get_torch_device


class MultiDimensionMSELoss(nn.Module):
    def __init__(self, num_classes=None):
        super(MultiDimensionMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes

    def forward(self, output, target):
        if target.dim() == 1 and self.num_classes:
            target = target.view((-1, self.num_classes))

        mse = self.mse(output, target)
        s = torch.mean(torch.sum(mse, dim=1))

        return s


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = torch.clamp(1 - target * output, min=0)
        return torch.mean(hinge_loss)


class ModelManager(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        criterion=HingeLoss(),
        device=get_torch_device(),
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

    def train(
        self,
        train_loader,
        eval_loader,
        epochs,
        loss_window=10,
        batch_size=None,
        num_classes=None,
    ):
        self.model.train()

        for epoch in range(epochs):
            batch_losses = []
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=True,
            )

            for batch in train_loader:
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

            if eval_loader:
                self.calc_metrics(eval_loader, "Test")
                print()

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

        print(f"Eval loss: {mean(batch_losses):.3f}")

    def calc_metrics(self, loader, label):
        self.model.eval()

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch)
                bsz = batch.num_graphs
                all_targets.extend(batch.y.view((bsz, -1)).cpu().numpy())
                all_predictions.extend(out.cpu().numpy())
        print(label)
        for i in range(4):
            print(i)
            print(
                f"\tR2: {r2_score([t[i] for t in all_targets], [p[i] for p in all_predictions]):.3f}"
            )
            corr, p = pearsonr(
                [t[i] for t in all_targets], [p[i] for p in all_predictions]
            )
            print(f"\tcorr: {corr: .2f}")
