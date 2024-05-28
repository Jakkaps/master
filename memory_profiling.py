import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader

from dialog_discrimination_dataset import DialogDiscriminationDataset
from model.dialog_discriminator import DialogDiscriminator
from model_manager import HingeLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DialogDiscriminator().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = HingeLoss()

dataset = DialogDiscriminationDataset(
    root="data/twitter_cs", dataset="twitter_cs", split="train"
)
loader = DataLoader(dataset, batch_size=25, shuffle=True)
input_data = next(iter(loader)).to(device)
target = input_data.y

scaler = GradScaler()


def check_mem_usage_mixed_precision():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with autocast():
        output = model(input_data)
        loss = criterion(output, target)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    current_memory = torch.cuda.memory_allocated(device) / (1024**2)

    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Current memory usage: {current_memory:.2f} MB")


def check_mem_usage():
    torch.cuda.empty_cache()
    output = model(input_data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    current_memory = torch.cuda.memory_allocated(device) / (1024**2)

    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Current memory usage: {current_memory:.2f} MB")


print("With mixed precision:")
check_mem_usage_mixed_precision()

print("\nWithout mixed precision:")
check_mem_usage()
