import logging
from typing import Callable, Dict, Tuple, Union

import pandas as pd
import surprise
from sklearn.model_selection import train_test_split

from utils.data import build_full_df


class Dataset:
    def __init__(
        self, trainset: surprise.Trainset, mode: Union["user", "item"] = "user"
    ):
        self.trainset = trainset
        self.mode = mode

    def user_ratings(self) -> Dict:
        return self.trainset.ur

    def item_ratings(self) -> Dict:
        return self.trainset.ir

    @property
    def n_features(self) -> int:
        return self.n_items if self.mode == "user" else self.n_users

    @property
    def n_items(self) -> int:
        return self.trainset.n_items

    @property
    def n_users(self) -> int:
        return self.trainset.n_users

    @property
    def n_ratings(self) -> int:
        return self.trainset.n_ratings

    def tvt(self, mode: Union["item", "user"] = "item") -> Tuple[Dict, Dict, Dict]:
        if mode == "user":
            ratings = self.user_ratings()
            n_features = self.n_items
        elif mode == "item":
            ratings = self.item_ratings()
            n_features = self.n_users
        else:
            raise ValueError

        train, test = train_test_split(ratings, test_size=0.2)
        valid, test = train_test_split(test, test_size=0.5)

        return train, valid, test


class DatasetFactory:
    """
    This class represents a factory that is responsible for generating datasets

    """

    def __init__(
        self,
        ratings_path: str = None,
        items_path: str = None,
        users_path: str = None,
        sep: str = ",",
    ):
        """
        Parameters
        ----------
        ratings_path : str
            path to the ratings file containing information about user item interaction, by default None
        items_path : str, optional
            path to the items file containing information about items, by default None
        users_path : str, optional
            path to the items file containing information about users, by default None
        sep : str, optional
            character or sequence of characters used to separate values, by default ","
        """

        # read data from csv files and join them into a single dataframe
        self.df = self.read_csv(ratings_path, items_path, users_path, sep)

    def read_csv(self, ratings_path, items_path, users_path, sep) -> pd.DataFrame:
        ratings_df = None
        users_df = None
        items_df = None

        logger = logging.getLogger("core.data")

        if ratings_path is not None:
            try:
                ratings_df = pd.read_csv(
                    ratings_path,
                    sep=sep,
                    engine="python",
                    encoding="latin-1",
                )
            except:
                logger.error("Failed to load ratings file")

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
                logger.error("Failed to load items file")

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
                logger.error("Failed to load users file")

        return build_full_df(ratings_df, users_df, items_df)

    def create_dataset(
        self,
        transformations: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        filters: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    ) -> Dataset:
        """
        Filters and transforms a copy of the underlying dataframe, and returns a Dataset object

        Parameters
        ----------
        transformations : Callable, optional
            Transformations to be applied to the data, by default lambda*args:args
        filters : Callable, optional
            Filters to be applied after the transformations, by default lambda*args:args

        Returns
        -------
        Dataset
            Instance of a Dataset Class
        """
        df = filters(transformations(self.df))
        df = df.loc[:, ["user", "item", "rating"]]
        trainset = surprise.Dataset.load_from_df(
            df, surprise.Reader(line_format="user item rating")
        ).build_full_trainset()
        dataset = Dataset(trainset)
        return dataset
