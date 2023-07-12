import pandas as pd
import pytest


@pytest.fixture
def ratings_df():
    df = pd.read_csv(
        "data/ml-1m/ratings.csv",
        sep=",",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )

    return df


@pytest.fixture
def items_df():
    df = pd.read_csv(
        "data/ml-1m/movies.csv",
        sep="::",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )

    return df


def users_df():
    df = pd.read_csv(
        "data/ml-1m/users.csv",
        sep=",",
        engine="python",
        encoding="latin-1",
        low_memory=True,
    )

    return df
