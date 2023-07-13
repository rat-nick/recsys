from typing import Callable, Dict, Tuple, Union

import pandas as pd
import surprise
from sklearn.model_selection import train_test_split

from utils.data import build_full_df, str_to_timestamp


class Dataset:
    def __init__(
        self,
        ratings_path: str = None,
        items_path: str = None,
        users_path: str = None,
        sep: str = ",",
        filter: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
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
        ratings_df = None
        users_df = None
        items_df = None

        # load all data into their respective dataframes
        if ratings_path is not None:
            try:
                ratings_df = pd.read_csv(
                    ratings_path,
                    sep=sep,
                    engine="python",
                    encoding="latin-1",
                    low_memory=True,
                )
            except:
                pass

        # lets assume that the ratings_df is mandatory and the others are optional
        if items_path is not None:
            try:
                items_df = pd.read_csv(
                    items_path,
                    sep=sep,
                    engine="python",
                    encoding="latin-1",
                    low_memory=True,
                )
            except:
                pass

        if users_path is not None:
            try:
                users_df = pd.read_csv(
                    users_path,
                    sep=sep,
                    engine="python",
                    encoding="latin-1",
                    low_memory=True,
                )
            except:
                pass

        self.filter = filter

        self.df = build_full_df(ratings_df, users_df, items_df)
        self.df = str_to_timestamp(self.df)

    @property
    def trainset(self) -> surprise.Trainset:
        if hasattr(self, "_trainset"):
            return self._trainset

        self.apply_filter()

        dataset = self.df.loc[:, ["user", "item", "rating"]]

        dataset = surprise.Dataset.load_from_df(
            dataset, surprise.Reader(line_format="user item rating")
        )
        # 4: build surprise trainset
        self._trainset = dataset.build_full_trainset()

        self.n_items = self._trainset.n_items
        self.n_users = self._trainset.n_users

        return self._trainset

    def apply_filter(self):
        self.df = self.filter(self.df)

    def train_valid_test_split(
        self,
        mode: Union["user", "item"] = "user",
    ) -> Tuple[Dict, Dict, Dict]:
        # we select what are going to be the training cases
        if mode == "user":
            ratings = self.trainset.ur
        elif mode == "item":
            ratings = self.trainset.ir
        else:
            raise ValueError

        train, test = train_test_split(ratings, test_size=0.2)
        valid, test = train_test_split(test, test_size=0.5)
        return train, valid, test
