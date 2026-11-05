import torch
import matplotlib.pyplot as plt

def get_device():
    """Select an available device, gpu is the first option, support cuda, mps and cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def synchronize_device(device):
    """Wait for pending device operations."""
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def plot_history(history, ax=None):
    """Plot training loss and training and validation accuracy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.clear()
    epochs = [item["epoch"] for item in history]

    ax.plot(
        epochs,
        [item["train"]["loss"] for item in history],
        linestyle="-",
        label="Train loss",
    )

    for split, name in [("train", "Train"), ("val", "Validation")]:
        ax.plot(
            epochs,
            [item[split]["accuracy"] for item in history],
            linestyle="--",
            label=f"{name} accuracy",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()

    return ax
