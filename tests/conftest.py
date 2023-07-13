import pandas as pd
import pytest

from core.dataset import Dataset
from utils.data import build_full_df, str_to_timestamp


@pytest.fixture
def full_dataset() -> Dataset:
    ds = Dataset(
        ratings_path="data/ml-100k/u.data",
        items_path="data/ml-100k/u.item",
        users_path="data/ml-100k/u.user",
        sep="|",
    )

    return ds


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
        encoding="latin-1",
        low_memory=True,
    )


@pytest.fixture
def ratings_df() -> pd.DataFrame:
    return pd.read_csv(
        "data/ml-100k/u.data",
        sep="|",
        encoding="latin-1",
        low_memory=True,
    )


@pytest.fixture
def full_df(ratings_df, users_df, items_df) -> pd.DataFrame:
    df = build_full_df(ratings_df, users_df, items_df)
    df = str_to_timestamp(df)
    return df
