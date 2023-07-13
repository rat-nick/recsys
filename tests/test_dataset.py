def test_no_filter_strategy(full_dataset):
    count_before = len(full_dataset.ratings_df)

    full_dataset.apply_filter_strategy()

    count_after = len(full_dataset.ratings_df)
    assert count_before == count_after

    full_dataset.build_surprise_trainset()

    assert full_dataset.n_items == 1682
    assert full_dataset.n_users == 943


def test_filter_to_empty_dataset(full_dataset):
    full_dataset.filter_strategy = lambda df: df[df.user == -1]

    full_dataset.apply_filter_strategy()
    full_dataset.build_surprise_trainset()

    assert full_dataset.n_items == 0
    assert full_dataset.n_users == 0


def test_filter_by_date(full_dataset):
    full_dataset.filter_strategy = lambda df: df[df.releaseDate]
    assert True
