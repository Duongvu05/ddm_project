"""Evaluate all models and print a comparison table.

Usage
-----
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --config configs/default.yaml --sample 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from ddm_project.baselines import (
    AgeSegmentedPopularityRecommender,
    BaselineConfig,
    GlobalPopularityRecommender,
    RecentPopularityRecommender,
    RepurchaseRecommender,
    train_test_split,
)
from ddm_project.cf_model import CFConfig, ItemCFRecommender
from ddm_project.model import ModelConfig, TwoStageLGBMRanker, load_data, make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all recommendation models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    data_dir = Path(raw.get("data", {}).get("data_dir", "data"))
    sample = args.sample

    logger.info("Loading data …")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    customers = pd.read_parquet(data_dir / "customers.parquet")
    articles = pd.read_parquet(data_dir / "articles.parquet")

    train_b, _, test_gt_b = train_test_split(tx)
    train_full, train_feat, val_tx, _, test_gt = make_splits(tx)

    results: dict[str, float] = {}

    # Baselines
    bl_cfg = BaselineConfig(**raw.get("baselines", {}))
    for model in [
        GlobalPopularityRecommender(),
        RecentPopularityRecommender(recent_weeks=bl_cfg.recent_weeks),
        RepurchaseRecommender(),
        AgeSegmentedPopularityRecommender(age_bins=bl_cfg.age_bins, age_labels=bl_cfg.age_labels),
    ]:
        if model.name == "age_segmented_popularity":
            model.fit(train_b, customers=customers)
        else:
            model.fit(train_b)
        results[model.name] = model.evaluate(test_gt_b, k=bl_cfg.k, sample=sample)

    # Item CF
    cf_cfg = CFConfig(**raw.get("cf", {}))
    cf = ItemCFRecommender(cf_cfg)
    cf.fit(train_b)
    results["item_cf"] = cf.evaluate(test_gt_b, k=cf_cfg.k, sample=sample)

    # Two-Stage LGBM
    m_cfg = ModelConfig(**raw.get("model", {}))
    m_cfg.data_dir = data_dir
    lgbm = TwoStageLGBMRanker(m_cfg)
    lgbm.fit(train_feat, val_tx, customers, articles, train_full)
    results["two_stage_lgbm"] = lgbm.evaluate(test_gt, k=m_cfg.k, sample=sample)

    # Print table
    logger.info("\n" + "=" * 52)
    logger.info(f"{'Model':<40} {'MAP@12':>8}")
    logger.info("=" * 52)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◄" if name == "two_stage_lgbm" else ""
        logger.info(f"{name:<40} {score:>8.6f}{marker}")
    logger.info("=" * 52)


if __name__ == "__main__":
    main()
