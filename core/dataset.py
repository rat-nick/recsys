from typing import Callable, List

import pandas as pd
import surprise


class Dataset:
    def __init__(
        self,
        ratings_path: str = None,
        items_path: str = None,
        users_path: str = None,
        sep: str = ",",
        filter_strategy: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    ):
        """
        This class represents a way to load a surprise dataset and build the full trainset with additional filtering applied

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

        self.filter_strategy = filter_strategy

    def build_surprise_trainset(self):

        dataset = self.ratings_df.loc[:, ["user", "item", "rating"]]

        dataset = surprise.Dataset.load_from_df(
            dataset, surprise.Reader(line_format="user item rating")
        )
        # 4: build surprise trainset
        trainset = dataset.build_full_trainset()

        self.n_items = trainset.n_items
        self.n_users = trainset.n_users

        return trainset

    def apply_filter_strategy(self):
        self.ratings_df = self.filter_strategy(self.ratings_df)
