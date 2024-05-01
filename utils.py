import torch


def get_torch_device() -> torch.device:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if device == "cuda":
        torch.cuda.empty_cache()  # Clear memory cache on the GPU if available
    elif device == "mps":
        torch.mps.empty_cache()

    return device


def get_base_filename(
    lr: float,
    epochs: int,
    batch_size: int,
    n_training_points: int,
    n_layers: int,
    graph_out_dim: int,
) -> str:
    return f"n_layers={n_layers}_lr={lr}_epochs={epochs}_batch_size={batch_size}_n_training_points={n_training_points}_graph_out_dim={graph_out_dim}"
