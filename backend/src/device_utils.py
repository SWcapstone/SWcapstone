import torch

def get_device():
    """
    Returns the best available device: CUDA for Windows/Linux, MPS for Mac, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
