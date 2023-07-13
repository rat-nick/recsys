import pandas as pd
import pytest
from core.dataset import Dataset


@pytest.fixture
def full_dataset() -> Dataset:
    ds = Dataset(
        ratings_path="data/ml-100k/u.data",
        items_path="data/ml-100k/u.item",
        users_path="data/ml-100k/u.user",
        sep="|",
    )

    return ds
