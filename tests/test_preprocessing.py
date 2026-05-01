"""Tests for preprocessing utilities."""

import pandas as pd
import pytest
from ddm_project.preprocessing import (
    add_week_index,
    find_obsolete_articles,
    reduce_memory,
    train_val_test_split,
)


def make_tx(dates: list[str], article_ids: list[str] | None = None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "t_dat": pd.to_datetime(dates),
        "customer_id": [f"c{i}" for i in range(n)],
        "article_id": article_ids or [f"a{i}" for i in range(n)],
        "price": [0.01] * n,
        "sales_channel_id": [2] * n,
    })


def test_add_week_index_latest_is_zero():
    tx = make_tx(["2020-09-22", "2020-09-15", "2020-09-08"])
    tx = add_week_index(tx)
    assert tx.loc[tx["t_dat"] == pd.Timestamp("2020-09-22"), "week"].values[0] == 0
    assert tx.loc[tx["t_dat"] == pd.Timestamp("2020-09-15"), "week"].values[0] == 1


def test_reduce_memory_dtypes():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1, 2]}, dtype="float64")
    df["b"] = df["b"].astype("int64")
    reduced = reduce_memory(df)
    assert reduced["a"].dtype == "float32"
    assert reduced["b"].dtype == "int32"


def test_find_obsolete():
    # article "old" has 100% sales before 2019 → obsolete
    # article "new" has 0% sales before 2019 → not obsolete
    dates = ["2018-06-01"] * 10 + ["2020-06-01"] * 10
    articles = ["old"] * 10 + ["new"] * 10
    tx = make_tx(dates, articles)
    obsolete = find_obsolete_articles(tx, cutoff_date="2019-01-01", threshold=0.95)
    assert "old" in obsolete
    assert "new" not in obsolete


def test_train_val_test_split_sizes():
    # Use weekly spaced dates so each week index has at least one row
    base = pd.Timestamp("2020-09-22")
    dates = [(base - pd.Timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(12)] * 2
    tx = make_tx(dates)
    tx = add_week_index(tx)
    train, val, test = train_val_test_split(tx, train_weeks=8, val_weeks=1, test_weeks=1)
    assert len(test) > 0
    assert len(val) > 0
    assert len(train) > 0
    # No overlap between splits
    test_weeks = set(test["week"])
    val_weeks = set(val["week"])
    train_weeks_set = set(train["week"])
    assert len(test_weeks & val_weeks) == 0
    assert len(val_weeks & train_weeks_set) == 0
