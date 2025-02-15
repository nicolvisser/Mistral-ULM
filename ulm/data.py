from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TokenizedItem:
    ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.ids.device

    @property
    def src_ids(self) -> torch.Tensor:
        return self.ids[:-1]

    @property
    def tgt_ids(self) -> torch.Tensor:
        return self.ids[1:]

    @property
    def loss_mask(self) -> torch.Tensor:
        return torch.arange(len(self.src_ids)) > 0

    @property
    def input_len(self) -> int:
        return len(self.src_ids)


@dataclass
class TokenizedBatch:
    src_ids: torch.Tensor  # (N,)
    tgt_ids: torch.Tensor  # (N,)
    loss_mask: torch.Tensor  # (N,)
    seqlens: List[int]  # (B,) where sum(seqlens) == N

    @property
    def batch_size(self) -> int:
        return len(self.seqlens)

    @property
    def device(self) -> torch.device:
        return self.src_ids.device

    def to(self, device: torch.device) -> "TokenizedBatch":
        return TokenizedBatch(
            src_ids=self.src_ids.to(device),
            tgt_ids=self.tgt_ids.to(device),
            loss_mask=self.loss_mask.to(device),
            seqlens=self.seqlens,
        )

    def __post_init__(self):
        assert self.src_ids.shape[0] == self.tgt_ids.shape[0] == self.loss_mask.shape[0]
        assert sum(self.seqlens) == self.src_ids.shape[0]


def collate_fn(items: List[TokenizedItem]) -> TokenizedBatch:
    return TokenizedBatch(
        src_ids=torch.cat([item.src_ids for item in items], dim=0),
        tgt_ids=torch.cat([item.tgt_ids for item in items], dim=0),
        loss_mask=torch.cat([item.loss_mask for item in items], dim=0),
        seqlens=[item.input_len for item in items],
    )
