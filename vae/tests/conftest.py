import pytest

from vae.data import *


@pytest.fixture
def dataset() -> Dataset:
    return create_df(
        "data/ml-100k/u.data",
        "data/ml-100k/u.item",
        "data/ml-100k/u.user",
        sep="|",
    ).create_dataset(mode="user")
