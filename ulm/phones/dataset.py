from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import TokenizedItem
from .tokenizer import PhonemeTokenizer


class TokenizedPhonemesDataset(Dataset):
    def __init__(
        self,
        strings_dir: str,
        pattern: str,
        tokenizer: PhonemeTokenizer,
    ):
        self.string_paths = sorted(list(Path(strings_dir).glob(pattern)))
        assert len(self.string_paths) > 0, "No strings found"
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.string_paths)

    def __getitem__(self, idx: int) -> str:
        path = self.string_paths[idx]
        chapter_id = path.parent.name
        string_value = path.read_text()
        ids = self.tokenizer.encode(string_value)
        return chapter_id, ids


class TokenizedPhonemesUtteranceDataset(Dataset):
    def __init__(
        self,
        strings_dir: str,
        pattern: str,
        tokenizer: PhonemeTokenizer,
    ):
        self.dataset = TokenizedPhonemesDataset(
            strings_dir=strings_dir, pattern=pattern, tokenizer=tokenizer
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        _, ids = self.dataset[idx]
        return TokenizedItem(ids=ids)


class TokenizedPhonemesChunkedDataset(Dataset):
    def __init__(
        self,
        strings_dir: str,
        pattern: str,
        tokenizer: PhonemeTokenizer,
        max_chunk_size: int,
    ):
        self.dataset = TokenizedPhonemesDataset(
            strings_dir=strings_dir, pattern=pattern, tokenizer=tokenizer
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
