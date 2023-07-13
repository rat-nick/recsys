from typing import Callable

import pandas as pd


def at_least_rated(n: int):
    return lambda df: df.groupby("user").filter(lambda x: len(x) >= n)


def at_most_rated(n: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df.groupby("user").filter(lambda x: len(x) <= n)


def is_genre(genre: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df[df[genre.lower()] == 1]


def is_not_genre(genre: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df[df[genre.lower()] == 0]


def released_before(date: pd.Timestamp) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df[df["releaseDate"] < date]


def released_after(date: pd.Timestamp) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df[df["releaseDate"] > date]


def released_between(
    start: pd.Timestamp, end: pd.Timestamp
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: released_after(start)(released_before(end)(df))
