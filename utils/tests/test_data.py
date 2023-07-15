from utils.data import *
import numpy as np


def test_str_to_datetime(full_df):
    df = str_to_datetime(full_df)
    # get the types of all the columns that end with date
    types = [df[col].dtype for col in df if col.lower().endswith("date")]
    assert all(t == np.dtype("datetime64[ns]") for t in types)
    assert len(full_df) == len(df)


def test_build_full_dataset(ratings_df, users_df, items_df):
    full_df = build_full_df(ratings_df, users_df, items_df)
    assert len(ratings_df) == len(full_df)
