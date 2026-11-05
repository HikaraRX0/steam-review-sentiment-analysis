import torch

def get_device():
    """Select an available device, gpu is the first option, support cuda, mps and cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
