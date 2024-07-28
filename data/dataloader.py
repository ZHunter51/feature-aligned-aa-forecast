import torch
from torch.utils import data as dt


class InfiniteDataLoader:
    def __init__(self, dataset: dt.Subset, batch_size: int, num_workers: int):
        batch_sampler = InfiniteSampler(
            dt.BatchSampler(
                dt.RandomSampler(dataset, replacement=True, num_samples=batch_size),
                batch_size,
                drop_last=True,
            )
        )
        self.iterator = iter(
            dt.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
            )
        )

    def __iter__(self):
        while True:
            yield next(self.iterator)

    def __len__(self) -> float:
        return torch.inf


class InfiniteSampler:
    def __init__(self, sampler: dt.Sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
