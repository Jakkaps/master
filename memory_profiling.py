import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from dialog_discrimination_dataset import DialogDiscriminationDataset
from model.dialog_discriminator import DialogDiscriminator
from model_manager import HingeLoss

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer, and input data
model = DialogDiscriminator().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = HingeLoss()

dataset = DialogDiscriminationDataset(
    root="data/twitter_cs", dataset="twitter_cs", split="train"
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
input_data = next(iter(loader)).to(device)
target = input_data.y


# Function to measure memory usage
def check_memory_usage():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    output = model(input_data)
    loss = criterion(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Retrieve the memory stats
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # Convert to MB
    current_memory = torch.cuda.memory_allocated(device) / (1024**2)  # Convert to MB

    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Current memory usage: {current_memory:.2f} MB")


# Call the function to check memory usage
check_memory_usage()
