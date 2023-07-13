def test_df_fixtures(users_df, items_df, ratings_df):
    assert len(users_df) != 0
    assert len(items_df) != 0
    assert len(ratings_df) != 0

def test_dataset