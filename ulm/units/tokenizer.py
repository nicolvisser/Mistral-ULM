from typing import List, Union

import numpy as np
import torch


class UnitTokenizer:
    """
    A tokenizer that adds a Beginning-Of-Sequence (BOS) token to the start of a sequence
    and shifts the unit IDs by 1.
    """

    BOS_id: int = 0

    def encode(self, units: List[int]) -> torch.Tensor:
        ids = torch.cat(
            [
                torch.tensor([self.BOS_id]),
                torch.tensor(units) + 1,
            ],
            dim=0,
        )
        return ids

    def decode(self, ids: torch.Tensor) -> List[int]:
        ids = ids[ids != self.BOS_id]
        units = ids - 1
        return units.tolist()
