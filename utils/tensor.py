from typing import Tuple

import torch
from sklearn.model_selection import train_test_split


def split(
    t: torch.Tensor, ratio: float = 0.2, nonzero_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    # all_but_first = lambda x: tuple([t for t in range(1, len(x.shape))])
    # reduce_to_first = lambda x: x.sum(dim=all_but_first(x))

    idx = t.nonzero()
    train_idx, test_idx = train_test_split(
        idx, test_size=ratio, random_state=42, shuffle=True
    )
    train = torch.zeros_like(t)
    idx = train_idx
    train[idx[:, 0], idx[:, 1]] = t[idx[:, 0], idx[:, 1]]

    test = torch.zeros_like(t)
    idx = test_idx
    test[idx[:, 0], idx[:, 1]] = t[idx[:, 0], idx[:, 1]]
    return train, test
