import torch

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def autocast_context(device):
    # MPS AMP support varies by torch release; float32 is the safe default.
    return torch.autocast(device_type="cpu", enabled=False)
