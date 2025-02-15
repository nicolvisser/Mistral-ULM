from itertools import groupby
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import TokenizedItem
from .tokenizer import UnitTokenizer


class TokenizedUnitsDataset(Dataset):
    def __init__(self, units_dir: str, pattern: str, tokenizer: UnitTokenizer):
        self.units_paths = sorted(list(Path(units_dir).glob(pattern)))
        assert len(self.units_paths) > 0, "No units found"
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.units_paths)

    def __getitem__(self, idx: int) -> List[int]:
        units_path = self.units_paths[idx]
        chapter_id = units_path.parent.name
        units = np.load(self.units_paths[idx])
        units_deduped = [k for k, _ in groupby(units)]
        ids = self.tokenizer.encode(units_deduped)
        return chapter_id, ids


class TokenizedUnitsUtteranceDataset(Dataset):
    def __init__(self, units_dir: str, pattern: str, tokenizer: UnitTokenizer):
        self.dataset = TokenizedUnitsDataset(
            units_dir=units_dir, pattern=pattern, tokenizer=tokenizer
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
    ):
        self.dataset = TokenizedUnitsDataset(
            units_dir=units_dir, pattern=pattern, tokenizer=tokenizer
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
