import logging
from typing import List

import pandas as pd
import surprise

logger = logging.getLogger("data.dataset")
logger.setLevel(logging.DEBUG)


class Dataset:
    def __init__(
        self,
        path: str = None,
        sep: str = ",",
        filter_strategies: List[function] = [],
    ):
        df = pd.read_csv(
            path, sep=sep, engine="python", encoding="latin-1", low_memory=True
        )
        logger.debug("Loaded pandas dataframe into memory")

        logger.debug("Applying filter strategies to dataset")

        # we want to apply all filter strategies
        for fs in filter_strategies:
            df = fs(df)

        logger.debug("Finished applying filter strategies to dataset")

        # remove all but the first 3 columns
        df = df.iloc[:, :3]
        logger.debug("Finished cleaning")

        self.dataset = surprise.Dataset.load_from_df(
            df, surprise.Reader(line_format="user item rating")
        )
        logger.info("Created surprise dataset")
        self.trainset = self.dataset.build_full_trainset()
        logger.info("Finished building surprise trainset")
