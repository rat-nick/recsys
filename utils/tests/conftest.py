import pandas as pd
import pytest

from utils.data import str_to_datetime
import torch


@pytest.fixture
def users_df() -> pd.DataFrame:
    return pd.read_csv(
        "data/ml-100k/u.user",
        sep="|",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )


@pytest.fixture
def items_df() -> pd.DataFrame:
    return pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )


@pytest.fixture
def ratings_df() -> pd.DataFrame:
    return pd.read_csv(
        "data/ml-100k/u.data",
        sep="|",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )


@pytest.fixture
def full_df() -> pd.DataFrame:
    return pd.read_csv("data/full_df.csv")


@pytest.fixture
def full_clean_df(full_df) -> pd.DataFrame:
    return str_to_datetime(full_df)


@pytest.fixture
def tensor1d() -> torch.Tensor:
    return torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


@pytest.fixture
def tensor2d() -> torch.Tensor:
    return torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
