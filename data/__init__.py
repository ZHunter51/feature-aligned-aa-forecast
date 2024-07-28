import pickle
from typing import List, Optional, Tuple

from torch import cuda
from torch.utils import data as dt

from .dataloader import InfiniteDataLoader
from .dataset import TimeSeriesDataset
from .utils import DATA_DIR

MODES = ["train", "valid", "test"]


def get_dataloaders(
    source_domains: List[str],
    target_domain: str,
    forecast_horizon: int,
    lookback_horizon: int,
    batch_size: int,
    dtype: str,
    fixed_data_size: Optional[int],
) -> Tuple[List[InfiniteDataLoader], List[dt.DataLoader], dt.DataLoader]:
    trainloaders, validloaders = [], []
    for domain in source_domains + [target_domain]:
        superdomain, domain = domain.split("/")
        cache_paths = {
            mode: DATA_DIR
            / superdomain
            / "cache"
            / f"{domain}_{lookback_horizon}x{forecast_horizon}_{mode}.pkl"
            for mode in MODES
        }
        if all(cache_paths[mode].exists() for mode in MODES):
            datasets = []
            for mode in MODES:
                with open(cache_paths[mode], "rb") as f:
                    datasets.append(pickle.load(f))
        else:
            dataset = TimeSeriesDataset(
                superdomain,
                domain,
                forecast_horizon,
                lookback_horizon,
                dtype,
                fixed_data_size,
            )
            datasets = dt.random_split(dataset, [0.7, 0.1, 0.2])
            for i, mode in enumerate(MODES):
                cache_paths[mode].parent.mkdir(parents=True, exist_ok=True)
                with open(cache_paths[mode], "wb") as f:
                    pickle.dump(datasets[i], f)
        trainloaders.append(
            InfiniteDataLoader(
                datasets[0], batch_size=batch_size, num_workers=cuda.device_count() * 4
            )
        )
        validloaders.append(
            dt.DataLoader(
                datasets[1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=cuda.device_count() * 4,
            )
        )
    _ = trainloaders.pop(), validloaders.pop()
    testloader = dt.DataLoader(
        datasets[2],
        batch_size=batch_size,
        shuffle=False,
        num_workers=cuda.device_count() * 4,
    )
    return trainloaders, validloaders, testloader
