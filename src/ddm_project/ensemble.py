"""Reciprocal Rank Fusion (RRF) ensemble over multiple recommenders.

RRF formula
-----------
    score(user, article) = Σ_model  weight_model / (k_rrf + rank_model(article))

where rank starts at 1 for the top prediction of each model.
Articles not in a model's list are ignored (score contribution = 0).

Why RRF?
--------
* Score-free — no need to normalise raw scores across models.
* Robust — dominated by items that rank highly in multiple models.
* Simple — no extra training needed.

Ensemble composition
--------------------
  1. TwoStageLGBMRanker  — holistic feature-based ranking  (weight 3)
  2. RepurchaseRecommender — strong recency/frequency signal  (weight 2)
  3. ItemCFRecommender     — collaborative "users like you"    (weight 1)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from ddm_project.baselines import map_at_k


# ── Protocol ──────────────────────────────────────────────────────────────────


class Recommender(Protocol):
    name: str

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]: ...


# ── Config ────────────────────────────────────────────────────────────────────


class EnsembleConfig(BaseModel):
    k_rrf: int = 60
    candidates_per_model: int = 50
    weights: list[float] = [3.0, 2.0, 1.0]   # lgbm, repurchase, item_cf
    sample_eval: int = 50_000


# ── RRF Ensemble ──────────────────────────────────────────────────────────────


class RRFEnsemble:
    """Reciprocal Rank Fusion over a list of fitted recommenders."""

    name = "rrf_ensemble"

    def __init__(
        self,
        models: list[Any],
        weights: list[float] | None = None,
        cfg: EnsembleConfig | None = None,
    ) -> None:
        self.models = models
        self.cfg = cfg or EnsembleConfig()
        self.weights = weights or self.cfg.weights
        if len(self.weights) != len(self.models):
            self.weights = [1.0] * len(self.models)

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        # Collect predictions from each model
        all_preds: list[dict[str, list[str]]] = []
        for model in self.models:
            preds = model.predict(customer_ids, k=self.cfg.candidates_per_model)
            all_preds.append(preds)
            logger.debug(f"  [{model.name}] predictions collected")

        # RRF fusion
        k_rrf = self.cfg.k_rrf
        final: dict[str, list[str]] = {}

        for uid in customer_ids:
            scores: dict[str, float] = {}
            for w, preds in zip(self.weights, all_preds):
                for rank, art in enumerate(preds.get(uid, []), start=1):
                    scores[art] = scores.get(art, 0.0) + w / (k_rrf + rank)
            top_k = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
            final[uid] = top_k

        return final

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

        logger.info(f"[{self.name}] Predicting for {len(gt_sub):,} users …")
        preds = self.predict(list(gt_sub.keys()), k=k)
        score = map_at_k(preds, gt_sub, k=k)
        logger.info(f"[{self.name}] MAP@{k} = {score:.6f}")
        return score


# ── Runner ────────────────────────────────────────────────────────────────────


LGBM_CACHE = "outputs/lgbm_model.pkl"


def _save_lgbm(model: Any, path: str = LGBM_CACHE) -> None:
    """Save only the lightweight parts of TwoStageLGBMRanker (no DataFrames)."""
    import pickle
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm_model": model.model,       # trained LGBMRanker (~MB)
        "obsolete": model._obsolete,      # set[str] ~10K items
        "cfg": model.cfg,                 # ModelConfig
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"LGBM model saved → {path}")


def _load_lgbm(path: str = LGBM_CACHE) -> Any | None:
    """Load lightweight payload; DataFrames are injected separately by caller."""
    import pickle
    from pathlib import Path
    if not Path(path).exists():
        return None
    logger.info(f"Loading cached LGBM model from {path} …")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    from ddm_project.model import TwoStageLGBMRanker
    shell = TwoStageLGBMRanker(payload["cfg"])
    shell.model = payload["lgbm_model"]
    shell._obsolete = payload["obsolete"]
    return shell


def run(cfg: EnsembleConfig | None = None, force_retrain: bool = False) -> dict[str, float]:
    if cfg is None:
        cfg = EnsembleConfig()

    from pathlib import Path

    from ddm_project.baselines import (
        RepurchaseRecommender,
        train_test_split,
    )
    from ddm_project.cf_model import CFConfig, ItemCFRecommender
    from ddm_project.model import ModelConfig, TwoStageLGBMRanker, load_data, make_splits

    m_cfg = ModelConfig(sample_eval=cfg.sample_eval)
    tx, customers, articles = load_data(m_cfg)
    train_full, train_feat, val_tx, _, test_gt = make_splits(tx)
    _, _, test_gt_b = train_test_split(tx)

    results: dict[str, float] = {}

    # ── LGBM: load cache or retrain ───────────────────────────────────────────
    lgbm = None if force_retrain else _load_lgbm()
    if lgbm is None:
        logger.info("Fitting TwoStageLGBMRanker (will be cached after) …")
        lgbm = TwoStageLGBMRanker(m_cfg)
        lgbm.fit(train_feat, val_tx, customers, articles, train_full)
        _save_lgbm(lgbm)
    else:
        # restore runtime state that isn't in the pickle
        lgbm._customers = customers
        lgbm._articles = articles
        lgbm._train_full = train_full
    results["two_stage_lgbm_v2"] = lgbm.evaluate(test_gt, k=12, sample=cfg.sample_eval)

    # ── Repurchase ────────────────────────────────────────────────────────────
    logger.info("Fitting RepurchaseRecommender …")
    repurchase = RepurchaseRecommender()
    repurchase.fit(train_full)
    results["repurchase"] = repurchase.evaluate(test_gt_b, k=12, sample=cfg.sample_eval)

    # ── Item CF ───────────────────────────────────────────────────────────────
    logger.info("Fitting ItemCFRecommender …")
    cf = ItemCFRecommender(CFConfig(sample_eval=cfg.sample_eval))
    cf.fit(train_full)
    results["item_cf"] = cf.evaluate(test_gt_b, k=12, sample=cfg.sample_eval)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    logger.info("Building RRF Ensemble …")
    ensemble = RRFEnsemble(models=[lgbm, repurchase, cf], weights=cfg.weights, cfg=cfg)
    results["rrf_ensemble"] = ensemble.evaluate(test_gt, k=12, sample=cfg.sample_eval)

    # ── Print table ───────────────────────────────────────────────────────────
    logger.info("─── Final Results ───")
    for name, s in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◄" if name == "rrf_ensemble" else ""
        logger.info(f"  {name:<40} MAP@12 = {s:.6f}{marker}")

    return results


if __name__ == "__main__":
    run()
