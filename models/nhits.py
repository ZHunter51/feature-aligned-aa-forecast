from math import ceil
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn.functional import interpolate

from .nbeats import LinearLayer


class NHiTS(nn.Module):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_horizon: int,
        num_stacks: int = 3,
        num_blocks: int = 4,
        num_layers: int = 4,
        activation: str = "ReLU",
        pooling: str = "MaxPool1d",
        interpolation: str = "linear",
        hidden_sizes: Union[List[int], int] = 512,
        kernel_size: int = 2,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lookback_horizon = lookback_horizon
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.activation = activation
        self.pooling = pooling
        self.interpolation = interpolation
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_stacks
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size

        self._build()

    def forward(
        self, data: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        residuals = data.flip(dims=(1,))
        forecast = data[:, -1:]
        features = []
        for block_idx, block in enumerate(self.blocks):
            backcast, block_forecast, hidden = block(residuals)

            if (block_idx + 1) % self.num_blocks == 0:
                features.append(hidden)

            residuals = residuals - backcast
            forecast = forecast + block_forecast
        if self.training:
            return forecast, features
        return forecast

    def _build(self):
        blocks = []
        for stack_idx in range(self.num_stacks):
            for _ in range(self.num_blocks):
                theta_size = self.lookback_horizon + self.forecast_horizon
                basis = BasisLayer(
                    backcast_size=self.lookback_horizon,
                    forecast_size=self.forecast_horizon,
                    interpolatation=self.interpolation,
                )
                block = NHiTSBlock(
                    input_size=self.lookback_horizon,
                    theta_size=theta_size,
                    num_layers=self.num_layers,
                    basis=basis,
                    pooling=self.pooling,
                    activation=self.activation,
                    hidden_size=self.hidden_sizes[stack_idx],
                    kernel_size=self.kernel_size,
                )
                blocks.append(block)
        self.blocks = nn.ModuleList(blocks)


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        num_layers: int,
        basis: nn.Module,
        activation: str,
        pooling: str,
        hidden_size: int,
        kernel_size: int,
    ):
        super().__init__()
        self.pool = getattr(nn, pooling)(
            kernel_size=kernel_size, stride=kernel_size, ceil_mode=True
        )
        self.extraction_layer = nn.Sequential(
            *[
                LinearLayer(
                    in_features=ceil(input_size / kernel_size),
                    out_features=hidden_size,
                    activation=activation,
                )
            ]
            + [
                LinearLayer(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    activation=activation,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.prediction_layer = nn.Linear(
            in_features=hidden_size, out_features=theta_size, bias=False
        )
        self.basis_layer = basis

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.pool(data)
        feature = self.extraction_layer(data)
        theta = self.prediction_layer(feature)
        backcast, forecast = self.basis_layer(theta)
        return backcast, forecast, feature


class BasisLayer(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolatation: str):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolatation = interpolatation

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, -self.forecast_size :]
        if self.interpolatation in ["nearest", "linear"]:
            knots = knots[:, None, :]
            forecast = interpolate(
                knots, size=self.forecast_size, mode=self.interpolatation
            )
            forecast = forecast[:, 0, :]
        elif self.interpolatation == "cubic":
            knots = knots[:, None, None, :]
            batch_size = backcast.shape[0]
            forecast = torch.zeros(
                (batch_size, self.forecast_size), device=knots.device
            )
            num_batch = ceil(knots.shape[0] / batch_size)
            for i in range(num_batch):
                forecast_i = interpolate(
                    knots[i * batch_size : (i + 1) * batch_size],
                    size=self.forecast_size,
                    mode="bicubic",
                )
                forecast[i * batch_size : (i + 1) * batch_size] += forecast_i[
                    :, 0, 0, :
                ]
        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpolatation}")
        return backcast, forecast
