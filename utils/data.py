import pandas as pd


def str_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        name = col.lower()
        if name.endswith("date"):
            df[col] = pd.to_datetime(df[col])

    return df


def build_full_df(
    ratings_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame
) -> pd.DataFrame:
    df = ratings_df
    if users_df is not None:
        df = df.join(users_df, on="user", rsuffix="(user)", how="left")
    if items_df is not None:
        df = df.join(items_df, on="item", rsuffix="(item)", how="left")
    return df
