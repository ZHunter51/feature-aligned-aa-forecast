import math
from typing import List, Tuple, Union

import torch
from torch import nn


class NBEATS(nn.Module):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_horizon: int,
        stack_types: List[str],
        share_weights: bool,
        num_blocks: int,
        num_layers: int,
        activation: str,
        hidden_sizes: Union[List[int], int],
        num_polynomials: int,
        num_harmonics: int,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lookback_horizon = lookback_horizon
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.activation = activation
        self.share_weights = share_weights
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * len(stack_types)
        self.hidden_sizes = hidden_sizes
        self.num_polynomials = num_polynomials
        self.num_harmonics = num_harmonics

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
        for stack_idx, stack_type in enumerate(self.stack_types):
            for block_idx in range(self.num_blocks):
                if self.share_weights and block_idx > 0:
                    block = blocks[-1]
                else:
                    if stack_type == "generic":
                        theta_size = self.lookback_horizon + self.forecast_horizon
                        basis = GenericBasis(
                            backcast_size=self.lookback_horizon,
                            forecast_size=self.forecast_horizon,
                        )
                    elif stack_type == "trend":
                        theta_size = 2 * (self.num_polynomials + 1)
                        basis = TrendBasis(
                            num_polynomials=self.num_polynomials,
                            backcast_size=self.lookback_horizon,
                            forecast_size=self.forecast_horizon,
                        )
                    elif stack_type == "seasonality":
                        theta_size = (
                            2
                            * 2
                            * (
                                math.ceil(
                                    self.num_harmonics / 2 * self.forecast_horizon
                                )
                                - (self.num_harmonics - 1)
                            )
                        )
                        basis = SeasonalityBasis(
                            num_harmonics=self.num_harmonics,
                            backcast_size=self.lookback_horizon,
                            forecast_size=self.forecast_horizon,
                        )
                    else:
                        raise ValueError(f"Unknown stack type: {stack_type}")
                    block = NBEATSBlock(
                        input_size=self.lookback_horizon,
                        theta_size=theta_size,
                        num_layers=self.num_layers,
                        basis=basis,
                        activation=self.activation,
                        hidden_size=self.hidden_sizes[stack_idx],
                    )
                blocks.append(block)
        self.blocks = nn.ModuleList(blocks)


class NBEATSg(NBEATS):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_horizon: int,
        share_weights: bool = True,
        num_blocks: int = 4,
        num_layers: int = 4,
        activation: str = "ReLU",
        hidden_sizes: Union[List[int], int] = 512,
        num_polynomials: int = 2,
        num_harmonics: int = 2,
    ):
        super().__init__(
            forecast_horizon,
            lookback_horizon,
            ["generic"] * 3,
            share_weights,
            num_blocks,
            num_layers,
            activation,
            hidden_sizes,
            num_polynomials,
            num_harmonics,
        )


class NBEATSi(NBEATS):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_horizon: int,
        share_weights: bool = True,
        num_blocks: int = 4,
        num_layers: int = 4,
        activation: str = "ReLU",
        hidden_sizes: Union[List[int], int] = [2048, 512, 512],
        num_polynomials: int = 2,
        num_harmonics: int = 2,
    ):
        super().__init__(
            forecast_horizon,
            lookback_horizon,
            ["trend", "seasonality", "seasonality"],
            share_weights,
            num_blocks,
            num_layers,
            activation,
            hidden_sizes,
            num_polynomials,
            num_harmonics,
        )


class NBEATSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        num_layers: int,
        basis: nn.Module,
        activation: str,
        hidden_size: int,
    ):
        super().__init__()
        self.extraction_layer = nn.Sequential(
            *[
                LinearLayer(
                    in_features=input_size,
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
        feature = self.extraction_layer(data)
        theta = self.prediction_layer(feature)
        backcast, forecast = self.basis_layer(theta)
        return backcast, forecast, feature


class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, -self.forecast_size :]
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(self, num_polynomials: int, backcast_size: int, forecast_size: int):
        super().__init__()
        polynomial_size = num_polynomials + 1
        self.backcast_basis = nn.Parameter(
            torch.cat(
                [
                    ((torch.arange(backcast_size) / backcast_size) ** i)[None, :]
                    for i in range(polynomial_size)
                ]
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            torch.cat(
                [
                    ((torch.arange(forecast_size) / forecast_size) ** i)[None, :]
                    for i in range(polynomial_size)
                ]
            ),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, polynomial_size:]
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bp,pt->bt", forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(self, num_harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        frequency = torch.cat(
            [
                torch.zeros(1),
                torch.arange(num_harmonics, num_harmonics / 2 * forecast_size)
                / num_harmonics,
            ]
        )[None, :]
        backcast_grid = (
            -2
            * torch.pi
            * (torch.arange(backcast_size)[:, None] / forecast_size)
            * frequency
        )
        forecast_grid = (
            2
            * torch.pi
            * (torch.arange(forecast_size)[:, None] / forecast_size)
            * frequency
        )

        backcast_cos_template = torch.cos(backcast_grid).transpose(1, 0)
        backcast_sin_template = torch.sin(backcast_grid).transpose(1, 0)
        backcast_template = torch.cat(
            [backcast_cos_template, backcast_sin_template], dim=0
        )

        forecast_cos_template = torch.cos(forecast_grid).transpose(1, 0)
        forecast_sin_template = torch.sin(forecast_grid).transpose(1, 0)
        forecast_template = torch.cat(
            [forecast_cos_template, forecast_sin_template], dim=0
        )

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, harmonic_size:]
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bp,pt->bt", forecast_theta, self.forecast_basis)
        return backcast, forecast


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))
