from itertools import groupby
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import TokenizedItem
from .tokenizer import UnitTokenizer

def simulate_markov(N, pi_F=0.008, a=0.15):
    rng = np.random.default_rng()
    # Compute b from stationary equation
    b = (pi_F * (1 - a)) / (1 - pi_F)

    seq = np.empty(N, dtype=np.uint8)  # 1 = F, 0 = S
    # initialize state according to stationary distribution
    seq[0] = rng.random() < pi_F
    for i in range(1, N):
        if seq[i-1] == 1:
            seq[i] = rng.random() < a
        else:
            seq[i] = rng.random() < b
    return seq

class TokenizedUnitsDataset(Dataset):
    def __init__(self, units_dir: str, pattern: str, tokenizer: UnitTokenizer, dedupe: bool = True):
        self.units_paths = sorted(list(Path(units_dir).glob(pattern)))
        assert len(self.units_paths) > 0, "No units found"
        self.tokenizer = tokenizer
        self.dedupe = dedupe

    def __len__(self) -> int:
        return len(self.units_paths)

    def __getitem__(self, idx: int) -> List[int]:
        units_path = self.units_paths[idx]
        chapter_id = units_path.parent.name
        units = np.load(self.units_paths[idx])
        random_mask = simulate_markov(len(units))
        # set units to 0 where random_mask is 1
        units = units * (1 - random_mask)
        ids = self.tokenizer.encode(units)
        if self.dedupe:
            ids = torch.unique_consecutive(ids)
        return chapter_id, ids


class TokenizedUnitsUtteranceDataset(Dataset):
    def __init__(self, units_dir: str, pattern: str, tokenizer: UnitTokenizer, dedupe: bool = True):
        self.dataset = TokenizedUnitsDataset(
            units_dir=units_dir, pattern=pattern, tokenizer=tokenizer, dedupe=dedupe
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> TokenizedItem:
        _, ids = self.dataset[idx]
        return TokenizedItem(ids=ids)


class TokenizedUnitsChunkedDataset(Dataset):
    def __init__(
        self,
        units_dir: str,
        pattern: str,
        tokenizer: UnitTokenizer,
        max_chunk_size: int,
        dedupe: bool = True
    ):
        self.dataset = TokenizedUnitsDataset(
            units_dir=units_dir, pattern=pattern, tokenizer=tokenizer, dedupe=dedupe
        )
        self.chunk_size = max_chunk_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> TokenizedItem:
        ids_list = []
        num_ids = 0

        # add first item with random starting point
        chapter_id, ids = self.dataset[idx]
        start_idx = np.random.randint(0, len(ids))
        ids_list.append(ids[start_idx:])
        num_ids += len(ids[start_idx:])
        idx += 1

        # if chunk size is not reached, try adding next items from the same chapter
        while num_ids < self.chunk_size:
            if idx >= len(self.dataset):
                break
            next_chapter_id, ids = self.dataset[idx]
            if next_chapter_id != chapter_id:
                break
            ids_list.append(ids)
            num_ids += len(ids)
            idx += 1

        ids_tensor = torch.cat(ids_list, dim=0)
        ids_tensor = ids_tensor[: self.chunk_size]
        return TokenizedItem(ids=ids_tensor)
