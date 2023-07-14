from typing import Tuple

import torch
from sklearn.model_selection import train_test_split


def split(
    t: torch.Tensor, ratio: float = 0.2, nonzero_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = t.nonzero()
    train_idx, test_idx = train_test_split(
        idx, test_size=ratio, random_state=42, shuffle=True
    )

    train = torch.zeros_like(t)
    test = torch.zeros_like(t)

    # Set nonzero values in train tensor
    train[tuple(train_idx.unbind(1))] = t[tuple(train_idx.unbind(1))]

    # Set nonzero values in test tensor
    test[tuple(test_idx.unbind(1))] = t[tuple(test_idx.unbind(1))]

    return train, test
