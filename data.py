"""Deterministic data helpers for v0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from world import Item, encode_item, STATES, make_item


@dataclass(frozen=True)
class HistoryConfig:
    name: str
    n_tasks: int = 384


class ItemDataset(Dataset):
    def __init__(self, items: Iterable[Item]):
        self.items = tuple(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return torch.tensor(encode_item(item), dtype=torch.long), torch.tensor(item.target_id, dtype=torch.long)


def repeat_to_count(items: tuple[Item, ...], n: int) -> tuple[Item, ...]:
    return tuple(items[i % len(items)] for i in range(n))


def history_A(cfg: HistoryConfig) -> tuple[Item, ...]:
    """Compositional pressure restricted to T1/T2; other ops are singleton-only."""
    pool = []
    for s in STATES:
        for a, b in (("T1", "T2"), ("T2", "T1"), ("T1", "T1"), ("T2", "T2")):
            pool.append(make_item((a, b), s))
        for op in ("T4", "T5"):
            pool.append(make_item((op,), s))
    return repeat_to_count(tuple(pool), cfg.n_tasks)


def history_B(cfg: HistoryConfig) -> tuple[Item, ...]:
    """Local/direct pressure with the same operator vocabulary and task count."""
    pool = []
    for s in STATES:
        for op in ("T1", "T2", "T4", "T5"):
            pool.append(make_item((op,), s))
        pool.append(make_item(("T1",), s))
        pool.append(make_item(("T2",), s))
    return repeat_to_count(tuple(pool), cfg.n_tasks)


def loader(items: Iterable[Item], batch_size: int = 64, shuffle: bool = False, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ItemDataset(items), batch_size=batch_size, shuffle=shuffle, generator=g)
