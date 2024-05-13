from statistics import mean

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm

from utils import get_torch_device


class MDCE(nn.Module):
    def __init__(self, num_classes=None):
        super(MDCE, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, output, target):
        target = target.view(-1, 2).to(torch.float32)

        return self.cross_entropy(output, target)


class MultiDimensionMSELoss(nn.Module):
    def __init__(self, num_classes=None):
        super(MultiDimensionMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes

    def forward(self, output, target, n_dims=4):
        if target.dim() == 1 and self.num_classes:
            target = target.view((-1, self.num_classes))

        se = torch.abs(output[:, :n_dims] - target[:, :n_dims])
        mse = torch.mean(se, dim=0)
        dim_mse = torch.sum(mse)

        return dim_mse


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
        model_base_name="",
        criterion=HingeLoss(),
        device=get_torch_device(),
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_base_path = model_base_name.split(".")[0]
        self.model_name = self.model_base_path.split("/")[-1]

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
        save_every_epoch=False,
        batch_size=None,
        num_classes=None,
        output_path="./output",
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

            if save_every_epoch:
                self.save(self.model_base_path + f"_epoch={epoch+1}.pth")

            if eval_loader:
                target, preds = self.eval(eval_loader)

                torch.save(
                    target,
                    f"{output_path}/{self.model_name}_targets_epoch={epoch+1}.pt",
                )
                torch.save(
                    preds,
                    f"{output_path}/{self.model_name}_predictions_epoch={epoch+1}.pt",
                )

                # if (epoch + 1) % 5 == 0:
                # self.calc_metrics(eval_loader, "Metric eval")

    def eval(self, loader, loss_window=10):
        self.model.eval()
        targets, preds = [], []

        batch_losses = []
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(loader), total=len(loader), desc="Evaluating", leave=True
            )
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch)
                loss = self.criterion(out, batch.y)

                targets.extend(batch.y.cpu().numpy())
                preds.extend(out.cpu().numpy())

                batch_losses.append(loss.item())
                progress_bar.set_postfix(
                    avg=mean(batch_losses),
                    window=mean(batch_losses[-loss_window:]),
                )
                progress_bar.update(1)

        return targets, preds

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

        for i in range(4):
            r_2 = r2_score([t[i] for t in all_targets], [p[i] for p in all_predictions])
            corr, p = pearsonr(
                [t[i] for t in all_targets], [p[i] for p in all_predictions]
            )

            print(f"\t Dim {i+1}:\tR^2: {r_2:.3f}\tCorr: {corr:.3f} (p={p:.2f})")
