"""Data preprocessing utilities: memory optimization, out-of-stock filtering, week index."""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    data_dir: Path = Path("data")
    obsolete_ratio_threshold: float = 0.95
    obsolete_cutoff_date: str = "2019-01-01"
    recent_weeks_for_train: int = 8


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64→float32 and int64→int32 to save ~40% RAM."""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def add_week_index(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add week column: 0 = most recent week, increasing into the past."""
    transactions = transactions.copy()
    transactions["week"] = (
        (transactions["t_dat"].max() - transactions["t_dat"]).dt.days // 7
    ).astype("int32")
    return transactions


def find_obsolete_articles(
    transactions: pd.DataFrame,
    cutoff_date: str = "2019-01-01",
    threshold: float = 0.95,
) -> set[str]:
    """Return article IDs where ≥threshold of all sales occurred before cutoff_date.

    These are likely discontinued items that should not be recommended.
    """
    tx = transactions.copy()
    tx["before_cutoff"] = tx["t_dat"] < cutoff_date

    total_sales = tx.groupby("article_id").size()
    old_sales = tx[tx["before_cutoff"]].groupby("article_id").size()

    ratio = (old_sales / total_sales).fillna(0)
    obsolete = set(ratio[ratio >= threshold].index.tolist())
    logger.info(
        f"Obsolete articles (≥{threshold:.0%} sales before {cutoff_date}): {len(obsolete):,}"
    )
    return obsolete


def preprocess_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers["age"] = customers["age"].fillna(customers["age"].median())
    customers["club_member_status"] = customers["club_member_status"].fillna("UNKNOWN")
    customers["fashion_news_frequency"] = customers["fashion_news_frequency"].fillna("NONE")
    return customers


def preprocess_articles(articles: pd.DataFrame) -> pd.DataFrame:
    articles = articles.copy()
    for col in articles.select_dtypes(include="object").columns:
        articles[col] = articles[col].fillna("UNKNOWN")
    return articles


def train_val_test_split(
    transactions: pd.DataFrame,
    train_weeks: int = 8,
    val_weeks: int = 1,
    test_weeks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Week-based split: test=most recent, val=next, train=next N weeks.

    Uses week index where 0 is the most recent week.
    """
    if "week" not in transactions.columns:
        transactions = add_week_index(transactions)

    test = transactions[transactions["week"] < test_weeks].copy()
    val = transactions[
        (transactions["week"] >= test_weeks)
        & (transactions["week"] < test_weeks + val_weeks)
    ].copy()
    train = transactions[
        (transactions["week"] >= test_weeks + val_weeks)
        & (transactions["week"] < test_weeks + val_weeks + train_weeks)
    ].copy()

    logger.info(
        f"Split — train: {len(train):,} rows ({train_weeks}w)  "
        f"val: {len(val):,} rows ({val_weeks}w)  "
        f"test: {len(test):,} rows ({test_weeks}w)"
    )
    return train, val, test


def run_preprocessing(cfg: PreprocessConfig | None = None) -> dict[str, pd.DataFrame]:
    """Full preprocessing pipeline. Returns dict with all processed DataFrames."""
    if cfg is None:
        cfg = PreprocessConfig()

    logger.info("Loading raw data …")
    articles = pd.read_parquet(cfg.data_dir / "articles.parquet")
    customers = pd.read_parquet(cfg.data_dir / "customers.parquet")
    transactions = pd.read_parquet(cfg.data_dir / "transactions.parquet")

    logger.info("Reducing memory …")
    customers = reduce_memory(customers)
    transactions = reduce_memory(transactions)

    logger.info("Cleaning …")
    customers = preprocess_customers(customers)
    articles = preprocess_articles(articles)
    transactions = add_week_index(transactions)

    logger.info("Detecting obsolete articles …")
    obsolete = find_obsolete_articles(
        transactions,
        cutoff_date=cfg.obsolete_cutoff_date,
        threshold=cfg.obsolete_ratio_threshold,
    )

    logger.info("Splitting …")
    train, val, test = train_val_test_split(
        transactions, train_weeks=cfg.recent_weeks_for_train
    )

    return {
        "articles": articles,
        "customers": customers,
        "transactions": transactions,
        "train": train,
        "val": val,
        "test": test,
        "obsolete_articles": pd.DataFrame({"article_id": list(obsolete)}),
    }
