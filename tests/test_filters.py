from utils.filters import *


def test_no_filters(full_df):
    filter = lambda x: x
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before == after


def test_users_rated_at_least_50(full_df):
    filter = at_least_rated(50)
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before > after
    assert after > 0


def test_users_rated_at_least_2000(full_df):
    filter = at_least_rated(2000)
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before > after
    assert after == 0


def test_users_rated_at_most_10(full_df):
    filter = at_most_rated(10)
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before > after
    assert after == 0


def test_users_rated_at_most_50(full_df):
    filter = at_most_rated(50)
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before > after
    assert after > 0


def test_released_before_1800(full_df):
    filter = released_before(pd.Timestamp(year=1800, month=1, day=1))
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before > after
    assert after == 0


def test_released_after_1800(full_df):
    filter = released_after(pd.Timestamp(year=1800, month=1, day=1))
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before >= after
    assert after > 0


def test_released_before_1990(full_df):
    filter = released_before(pd.Timestamp(year=1990, month=1, day=1))
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before >= after
    assert after > 0


def test_released_after_1990(full_df):
    filter = released_after(pd.Timestamp(year=1990, month=1, day=1))
    before = len(full_df)
    full_df = filter(full_df)
    after = len(full_df)
    assert before >= after
    assert after > 0
