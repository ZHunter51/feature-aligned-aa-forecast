from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils import data as dt

from .utils import DATA_DIR


class TimeSeriesDataset(dt.Dataset):
    def __init__(
        self,
        superdomain: str,
        domain: str,
        forecast_horizon: int,
        lookback_horizon: int,
        dtype: str,
        fixed_data_size: Optional[int],
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lookback_horizon = lookback_horizon
        time_range = forecast_horizon + lookback_horizon

        raw = pd.read_csv(DATA_DIR / superdomain / f"{domain}.csv")[
            ["time", "series", "value"]
        ]
        proc = pd.DataFrame(
            columns=["series", "value"] + [f"value{t}" for t in range(1, time_range)]
        )
        for _, data in raw.groupby("series"):
            proc = pd.concat(
                [
                    proc,
                    pd.concat(
                        [data]
                        + [
                            data["value"].shift(-t).rename(f"value{t}")
                            for t in range(1, time_range)
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        self.data = (
            proc.iloc[:, 2:]
            .replace(".", float("nan"))
            .dropna(axis=0)
            .astype(getattr(np, dtype))
        )
        if fixed_data_size:
            self.data = self.data.sample(fixed_data_size)
        assert len(self.data) > 0, f"Empty dataset for {superdomain}/{domain}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.data.iloc[idx, : self.lookback_horizon].values),
            torch.tensor(self.data.iloc[idx, -self.forecast_horizon :].values),
        )

    def __len__(self) -> int:
        return len(self.data)
