import click
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dialog_rating_dataset import DialogRatingDataset
from model.dialog_rater import DialogRater
from model_manager import MultiDimensionMSELoss
from utils import get_torch_device


@click.command()
@click.option("--epoch", default=1, type=int)
@click.option("--variant", default="", type=str)
@click.option("--n_iterations", default=100, type=int)
def main(epoch, variant, n_iterations, n_layers=10, graph_out_dim=10):
    model_name = f"n_layers={n_layers}_graph_out_dim={graph_out_dim}_epoch={epoch}.pth"
    dataset_name = "ratings"
    root = f"data/{dataset_name}"
    dataset = DialogRatingDataset(root=root, dataset=dataset_name)

    model_path = f"ckpts{variant}/{model_name}"
    device = get_torch_device()

    criterion = MultiDimensionMSELoss(num_classes=4)
    lr = 0.001
    epochs = 10
    batch_size = 10
    n_size = int(len(dataset) * 0.80)  # Example: 80% training size

    dim_corrs = []

    for _ in tqdm(range(n_iterations), desc="Bootstrap iterations"):
        train_indices = np.random.choice(len(dataset), size=n_size, replace=True)
        test_indices = list(set(range(len(dataset))) - set(train_indices))

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        state_dict = torch.load(model_path, map_location=device)
        graph_embed_state_dict = {
            k.replace("graph_embed.", ""): v
            for k, v in state_dict.items()
            if "graph_embed" in k
        }
        model = DialogRater(
            n_graph_layers=n_layers,
            graph_out_dim=graph_out_dim,
            n_hidden_layers=2,
            hidden_dim=128,
        )
        model.graph_embed.load_state_dict(graph_embed_state_dict)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            for batch in train_loader:
                batch_data = batch.to(device)
                y_pred = model(batch_data)
                loss = criterion(y_pred, batch_data.y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():

            ys = []
            y_preds = []

            for batch in test_loader:
                batch_data = batch.to(device)
                y_pred = model(batch_data)
                y = batch_data.y.view(-1, 4)

                ys.append(y)
                y_preds.append(y_pred)

            ys = torch.cat(ys, dim=0)
            y_preds = torch.cat(y_preds, dim=0)

            bootstrap_corrs = []
            for dim_idx in range(y_preds.shape[1]):
                bootstrap_corrs.append(
                    pearsonr(y_preds[:, dim_idx].cpu(), ys[:, dim_idx].cpu())[0]
                )

            dim_corrs.append(bootstrap_corrs)

    torch.save(dim_corrs, f"bootstrap_results/corrs{variant}.pt")


if __name__ == "__main__":
    main()
