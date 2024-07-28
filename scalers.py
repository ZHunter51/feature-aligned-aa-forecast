__all__ = ["softmax", "identity", "tanh"]

import torch
from torch.nn import functional as F


def safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    div = a / b
    div[(div.isnan()) + (div.isinf())] = 0
    return div


def softmax(tensor: torch.Tensor) -> torch.Tensor:
    return F.softmax(tensor, dim=1)


def identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def tanh(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tanh(tensor)
