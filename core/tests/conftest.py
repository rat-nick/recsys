import pytest

from core.data import *


@pytest.fixture
def ratings_path():
    return "data/ml-100k/u.data"


@pytest.fixture
def items_path():
    return "data/ml-100k/u.item"


@pytest.fixture
def users_path():
    return "data/ml-100k/u.user"


@pytest.fixture
def sep():
    return "|"


@pytest.fixture
def df_ratings(ratings_path, sep):
    df = create_df(
        ratings_path=ratings_path,
        sep=sep,
    )
    return df


@pytest.fixture
def df_full(ratings_path, items_path, users_path, sep):
    df = create_df(
        ratings_path=ratings_path,
        items_path=items_path,
        users_path=users_path,
        sep=sep,
    )
    return df


@pytest.fixture
def dataset_user_based(df_full):
    return create_dataset(df_full)


@pytest.fixture
def dataset_item_based(df_full):
    return create_dataset(df_full, mode="item")
