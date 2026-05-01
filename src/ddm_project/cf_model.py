"""Item-based Collaborative Filtering using sparse co-purchase similarity.

Algorithm
---------
1. Build a sparse user-item binary matrix from training transactions.
2. Compute item-item co-purchase counts:  C = X^T @ X  (sparse)
3. Normalise to cosine similarity:  sim(i,j) = C[i,j] / sqrt(C[i,i] * C[j,j])
4. For each target user:
   a. Collect their purchased items.
   b. Sum similarity scores across all candidate items.
   c. Zero out already-purchased items (optional).
   d. Return top-K.
5. Fall back to global recent popularity for cold-start users.

Practical limits
----------------
Full 2-year matrix is 1.36M × 104K — too large for dense similarity.
We restrict to the last `recent_weeks` weeks and only users with ≥ 2
purchases, which reduces the matrix to a manageable sparse form.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger
from pydantic import BaseModel, field_validator

from ddm_project.baselines import map_at_k


# ── Config ────────────────────────────────────────────────────────────────────


class CFConfig(BaseModel):
    data_dir: Path = Path("data")
    k: int = 12
    recent_weeks: int = 8
    min_item_support: int = 5
    top_similar: int = 50
    fallback_n: int = 100
    sample_eval: int = 50_000

    @field_validator("k")
    @classmethod
    def k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be positive")
        return v


# ── Model ─────────────────────────────────────────────────────────────────────


class ItemCFRecommender:
    """Item-item Collaborative Filtering with cosine similarity."""

    name = "item_cf"

    def __init__(self, cfg: CFConfig | None = None) -> None:
        self.cfg = cfg or CFConfig()
        self._item_sim: sp.csr_matrix | None = None
        self._item_index: dict[str, int] = {}
        self._index_item: dict[int, str] = {}
        self._user_items: dict[str, list[int]] = {}
        self._fallback: list[str] = []

    def fit(self, train: pd.DataFrame, **kwargs: Any) -> None:
        t0 = time.time()
        t_max = train["t_dat"].max()
        cutoff = t_max - pd.Timedelta(weeks=self.cfg.recent_weeks)
        recent = train[train["t_dat"] >= cutoff].copy()

        # Filter low-support items
        item_counts = recent["article_id"].value_counts()
        valid_items = item_counts[item_counts >= self.cfg.min_item_support].index
        recent = recent[recent["article_id"].isin(valid_items)]

        # Build index maps
        items = sorted(recent["article_id"].unique())
        users = sorted(recent["customer_id"].unique())
        self._item_index = {a: i for i, a in enumerate(items)}
        self._index_item = {i: a for a, i in self._item_index.items()}
        user_index = {u: i for i, u in enumerate(users)}

        n_users, n_items = len(users), len(items)
        logger.info(f"Building sparse matrix: {n_users:,} users × {n_items:,} items")

        row = recent["customer_id"].map(user_index).values
        col = recent["article_id"].map(self._item_index).values
        data = np.ones(len(recent), dtype=np.float32)

        X = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))
        X.data[:] = 1.0  # binarise

        # Co-purchase matrix  C = X^T @ X
        logger.info("Computing item-item co-purchase matrix …")
        C = (X.T @ X).toarray().astype(np.float32)

        # Cosine normalisation
        diag = np.sqrt(np.diag(C))
        diag[diag == 0] = 1.0
        C /= diag[:, None]
        C /= diag[None, :]
        np.fill_diagonal(C, 0.0)

        # Keep only top-K similar items per item (sparse)
        logger.info("Keeping top similar items per item …")
        top_k = self.cfg.top_similar
        for i in range(len(C)):
            row_vals = C[i]
            if len(row_vals) > top_k:
                threshold = np.partition(row_vals, -top_k)[-top_k]
                row_vals[row_vals < threshold] = 0.0
        self._item_sim = sp.csr_matrix(C)

        # Store user purchase history (as item indices)
        self._user_items = (
            recent.groupby("customer_id")["article_id"]
            .apply(lambda s: [self._item_index[a] for a in s if a in self._item_index])
            .to_dict()
        )

        # Global fallback (recent popular)
        self._fallback = (
            train[train["t_dat"] >= cutoff]["article_id"]
            .value_counts()
            .head(self.cfg.fallback_n)
            .index.tolist()
        )

        logger.info(f"[{self.name}] fitted in {time.time() - t0:.1f}s")

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        assert self._item_sim is not None, "Call fit() first"
        sim = self._item_sim.toarray()
        n_items = sim.shape[0]
        preds: dict[str, list[str]] = {}

        for uid in customer_ids:
            history = self._user_items.get(uid, [])

            if not history:
                preds[uid] = self._fallback[:k]
                continue

            # Score = sum of similarities to all purchased items
            scores = np.zeros(n_items, dtype=np.float32)
            for idx in history:
                scores += sim[idx]

            # Zero out already purchased
            for idx in history:
                scores[idx] = 0.0

            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            recs = [self._index_item[i] for i in top_indices if scores[i] > 0]

            # Fill with fallback if needed
            if len(recs) < k:
                purchased_set = {self._index_item[i] for i in history}
                extras = [a for a in self._fallback if a not in set(recs) and a not in purchased_set]
                recs = recs + extras[: k - len(recs)]

            preds[uid] = recs[:k]

        return preds

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


# ── Runner ────────────────────────────────────────────────────────────────────


def run(cfg: CFConfig | None = None) -> float:
    if cfg is None:
        cfg = CFConfig()

    tx = pd.read_parquet(cfg.data_dir / "transactions.parquet")
    last = tx["t_dat"].max()
    test_start = last - pd.Timedelta(days=6)
    train = tx[tx["t_dat"] < test_start].copy()
    test_tx = tx[tx["t_dat"] >= test_start].copy()
    test_gt: dict[str, set[str]] = (
        test_tx.groupby("customer_id")["article_id"].apply(set).to_dict()
    )

    model = ItemCFRecommender(cfg)
    model.fit(train)
    score = model.evaluate(test_gt, k=cfg.k, sample=cfg.sample_eval)
    return score


if __name__ == "__main__":
    run()
