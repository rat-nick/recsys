import logging
from typing import List

import pandas as pd
import surprise


class Dataset:
    def __init__(
        self,
        ratings_path: str = None,
        items_path: str = None,
        users_path: str = None,
        sep: str = ",",
        filter_strategies: List[function] = [],
    ):
        """This claass represents a way to load a surprise dataset and build the full trainset with additional filtering applied

        Parameters
        ----------
        ratings_path : str, optional
            path to the ratings file containing information about user item interaction, by default None
        items_path : str, optional
            path to the items file containing information about items, by default None
        users_path : str, optional
            path to the items file containing information about users, by default None
        sep : str, optional
            character or sequence of characters used to separate values, by default ","
        filter_strategies : List[function], optional
            list of functions to be applied to dataframe sequentialy, by default []
        """
        # load all data into their respective dataframes
        if ratings_path is not None:
            self.ratings_df = pd.read_csv(
                ratings_path,
                sep=sep,
                engine="python",
                encoding="latin-1",
                low_memory=True,
            )

        # lets assume that the ratings_df is mandatory and the others are optional
        if items_path is not None:
            items_df = pd.read_csv(
                items_path,
                sep=sep,
                engine="python",
                encoding="latin-1",
                low_memory=True,
            )
            self.ratings_df = self.ratings_df.join(
                items_df, on="item", how="left", rsuffix="item"
            )

        if users_path is not None:
            users_df = pd.read_csv(
                users_path,
                sep=sep,
                engine="python",
                encoding="latin-1",
                low_memory=True,
            )
            self.ratings_df = self.ratings_df.join(
                users_df, on="user", how="left", rsuffix="user"
            )
        # 2: perform filtering using filter strategies
        for fs in filter_strategies:
            self.ratings_df = fs(self.ratings_df)

        # 2.1: get only user, item and rating columns
        df = df.iloc[:, ["user", "item", "rating"]]

        # 3: build surprise dataset
        self.dataset = surprise.Dataset.load_from_df(
            df, surprise.Reader(line_format="user item rating")
        )
        # 4: build surprise trainset
        self.trainset = self.dataset.build_full_trainset()

    def build_dataset(self) -> surprise.Dataset:
        return surprise.Dataset.load_from_df()
