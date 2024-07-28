__all__ = ["SMAPE", "MAPE"]

import torch
from torch import nn


def safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    div = a / b
    div[(div.isnan()) + (div.isinf())] = 0
    return div


class SMAPE(nn.Module):
    def forward(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            safe_div(
                torch.abs(true - pred),
                (torch.abs(true) + torch.abs(pred)) / 2,
            )
        )


class MAPE(nn.Module):
    def forward(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(safe_div(true - pred, true)))
