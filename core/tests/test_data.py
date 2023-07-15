from core.data import *


def test_create_df_only_ratings(ratings_path, sep):
    df = create_df(
        ratings_path=ratings_path,
        sep=sep,
    )
    assert df is not None
    assert isinstance(df, pd.DataFrame)


def test_create_df_full(ratings_path, items_path, users_path, sep):
    df = create_df(
        ratings_path=ratings_path,
        items_path=items_path,
        users_path=users_path,
        sep=sep,
    )
    assert df is not None
    assert isinstance(df, pd.DataFrame)


def test_create_dataset_ratings_only(df_ratings):
    dataset = create_dataset(df=df_ratings)
    assert isinstance(dataset, Dataset)


def test_create_dataset_full(df_full):
    dataset = create_dataset(df=df_full)
    assert isinstance(dataset, Dataset)


def test_n_features_user_based(dataset_user_based):
    assert dataset_user_based.n_features == dataset_user_based.n_items


def test_n_featuers_item_based(dataset_item_based):
    assert dataset_item_based.n_features == dataset_item_based.n_users


def test_tvt_user_based(dataset_user_based):
    dataset = dataset_user_based
    train, valid, test = dataset.tvt()
    assert len(train) + len(test) + len(valid) == dataset.n_cases
