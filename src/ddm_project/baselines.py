"""Four recommendation baselines for H&M Fashion dataset.

Train / test split: last week of the dataset (2020-09-16..2020-09-22) is the
test set; everything before is train.

Metric: MAP@12  (Mean Average Precision at 12)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator


# ── Config ────────────────────────────────────────────────────────────────────


class BaselineConfig(BaseModel):
    data_dir: Path = Path("data")
    k: int = 12
    recent_weeks: int = 2
    age_bins: list[int] = [15, 25, 35, 45, 55, 65, 100]
    age_labels: list[str] = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    sample_eval: int = 50_000

    @field_validator("k")
    @classmethod
    def k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be positive")
        return v


# ── Metric ────────────────────────────────────────────────────────────────────


def _ap_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    """Average precision@k for a single user."""
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        if p in actual:
            hits += 1
            score += hits / i
    return score / min(len(actual), k) if actual else 0.0


def map_at_k(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int = 12,
) -> float:
    """Mean Average Precision@k over all users in ground_truth."""
    scores = [
        _ap_at_k(predictions.get(uid, []), gts, k)
        for uid, gts in ground_truth.items()
    ]
    return float(np.mean(scores))


# ── Data loading & split ──────────────────────────────────────────────────────


def load_data(cfg: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading parquet files …")
    transactions = pd.read_parquet(cfg.data_dir / "transactions.parquet")
    customers = pd.read_parquet(cfg.data_dir / "customers.parquet")
    articles = pd.read_parquet(cfg.data_dir / "articles.parquet")
    logger.info(
        f"Loaded – transactions: {len(transactions):,}, "
        f"customers: {len(customers):,}, articles: {len(articles):,}"
    )
    return transactions, customers, articles


def train_test_split(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    """Split last calendar week off as test; rest as train."""
    last_date = transactions["t_dat"].max()
    test_start = last_date - pd.Timedelta(days=6)

    train = transactions[transactions["t_dat"] < test_start].copy()
    test = transactions[transactions["t_dat"] >= test_start].copy()

    ground_truth: dict[str, set[str]] = (
        test.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    logger.info(
        f"Train: {len(train):,} rows | Test: {len(test):,} rows "
        f"({test_start.date()} – {last_date.date()}) | "
        f"Test users: {len(ground_truth):,}"
    )
    return train, test, ground_truth


# ── Base class ────────────────────────────────────────────────────────────────


class BaseRecommender:
    name: str = "base"

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        raise NotImplementedError

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        raise NotImplementedError

    def evaluate(
        self,
        ground_truth: dict[str, set[str]],
        k: int = 12,
        sample: int | None = None,
    ) -> float:
        if sample and sample < len(ground_truth):
            rng = np.random.default_rng(42)
            uids = rng.choice(list(ground_truth.keys()), size=sample, replace=False).tolist()
            gt_sub = {u: ground_truth[u] for u in uids}
        else:
            gt_sub = ground_truth

        preds = self.predict(list(gt_sub.keys()), k=k)
        score = map_at_k(preds, gt_sub, k=k)
        logger.info(f"[{self.name}] MAP@{k} = {score:.6f}")
        return score


# ── Baseline 1: Global Popularity ────────────────────────────────────────────


class GlobalPopularityRecommender(BaseRecommender):
    """Recommend the K globally most-purchased articles to every user."""

    name = "global_popularity"

    def __init__(self) -> None:
        self._top_articles: list[str] = []

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        self._top_articles = (
            train["article_id"]
            .value_counts()
            .head(100)
            .index.tolist()
        )
        logger.info(f"[{self.name}] fitted – top article: {self._top_articles[0]}")

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        recs = self._top_articles[:k]
        return {uid: recs for uid in customer_ids}


# ── Baseline 2: Recent Popularity (last N weeks) ──────────────────────────────


class RecentPopularityRecommender(BaseRecommender):
    """Recommend K articles most popular in the last *recent_weeks* weeks."""

    name = "recent_popularity"

    def __init__(self, recent_weeks: int = 2) -> None:
        self.recent_weeks = recent_weeks
        self._top_articles: list[str] = []

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        cutoff = train["t_dat"].max() - pd.Timedelta(weeks=self.recent_weeks)
        recent = train[train["t_dat"] >= cutoff]
        self._top_articles = (
            recent["article_id"]
            .value_counts()
            .head(100)
            .index.tolist()
        )
        logger.info(
            f"[{self.name}] fitted using last {self.recent_weeks} weeks "
            f"({len(recent):,} rows) – top: {self._top_articles[0]}"
        )

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        recs = self._top_articles[:k]
        return {uid: recs for uid in customer_ids}


# ── Baseline 3: Repurchase (personal history) ────────────────────────────────


class RepurchaseRecommender(BaseRecommender):
    """Recommend K articles the customer bought most recently.

    Falls back to global-popularity items when a customer has < k history.
    """

    name = "repurchase"

    def __init__(self) -> None:
        self._user_history: dict[str, list[str]] = {}
        self._fallback: list[str] = []

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        # Sort by date; keep unique articles in reverse-chronological order
        sorted_tx = train.sort_values("t_dat", ascending=False)
        self._user_history = (
            sorted_tx.groupby("customer_id")["article_id"]
            .apply(lambda s: list(dict.fromkeys(s.tolist())))  # deduplicate, preserve order
            .to_dict()
        )
        self._fallback = (
            train["article_id"].value_counts().head(100).index.tolist()
        )
        logger.info(f"[{self.name}] fitted – {len(self._user_history):,} user histories")

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        preds: dict[str, list[str]] = {}
        for uid in customer_ids:
            history = self._user_history.get(uid, [])[:k]
            if len(history) < k:
                # Fill with popular items not already in history
                extras = [a for a in self._fallback if a not in set(history)]
                history = history + extras[: k - len(history)]
            preds[uid] = history
        return preds


# ── Baseline 4: Age-Segmented Popularity ─────────────────────────────────────


class AgeSegmentedPopularityRecommender(BaseRecommender):
    """Recommend K articles most popular within the customer's age group.

    Age groups: 16-24, 25-34, 35-44, 45-54, 55-64, 65+
    Falls back to global popularity for customers without age data.
    """

    name = "age_segmented_popularity"

    def __init__(
        self,
        age_bins: list[int] | None = None,
        age_labels: list[str] | None = None,
    ) -> None:
        self.age_bins = age_bins or [15, 25, 35, 45, 55, 65, 100]
        self.age_labels = age_labels or ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        self._segment_tops: dict[str, list[str]] = {}
        self._global_top: list[str] = []
        self._customer_segment: dict[str, str] = {}

    def fit(
        self,
        train: pd.DataFrame,
        customers: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        if customers is None:
            raise ValueError("AgeSegmentedPopularityRecommender requires customers DataFrame")

        customers = customers.copy()
        customers["age_group"] = pd.cut(
            customers["age"],
            bins=self.age_bins,
            labels=self.age_labels,
            right=True,
        )
        self._customer_segment = (
            customers.dropna(subset=["age_group"])
            .set_index("customer_id")["age_group"]
            .astype(str)
            .to_dict()
        )

        tx_cust = train.merge(
            customers[["customer_id", "age_group"]].dropna(),
            on="customer_id",
            how="left",
        )

        for group, grp_df in tx_cust.groupby("age_group", observed=True):
            self._segment_tops[str(group)] = (
                grp_df["article_id"].value_counts().head(100).index.tolist()
            )

        self._global_top = train["article_id"].value_counts().head(100).index.tolist()
        logger.info(
            f"[{self.name}] fitted – {len(self._segment_tops)} segments: "
            + ", ".join(self._segment_tops.keys())
        )

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        preds: dict[str, list[str]] = {}
        for uid in customer_ids:
            seg = self._customer_segment.get(uid)
            top = self._segment_tops.get(seg, self._global_top) if seg else self._global_top
            preds[uid] = top[:k]
        return preds


# ── Runner ────────────────────────────────────────────────────────────────────


def run_all_baselines(cfg: BaselineConfig | None = None) -> dict[str, float]:
    """Fit and evaluate all 4 baselines; return MAP@K scores."""
    if cfg is None:
        cfg = BaselineConfig()

    transactions, customers, articles = load_data(cfg)
    train, _test, ground_truth = train_test_split(transactions)

    models: list[BaseRecommender] = [
        GlobalPopularityRecommender(),
        RecentPopularityRecommender(recent_weeks=cfg.recent_weeks),
        RepurchaseRecommender(),
        AgeSegmentedPopularityRecommender(
            age_bins=cfg.age_bins,
            age_labels=cfg.age_labels,
        ),
    ]

    results: dict[str, float] = {}
    for model in models:
        logger.info(f"─── Fitting {model.name} ───")
        if model.name == "age_segmented_popularity":
            model.fit(train, customers=customers)
        else:
            model.fit(train)

        score = model.evaluate(ground_truth, k=cfg.k, sample=cfg.sample_eval)
        results[model.name] = score

    logger.info("─── Results ───")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:<35} MAP@{cfg.k} = {score:.6f}")

    return results


if __name__ == "__main__":
    run_all_baselines()
