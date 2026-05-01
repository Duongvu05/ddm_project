"""Train the two-stage LightGBM ranker and save the model.

Usage
-----
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import yaml
from loguru import logger

from ddm_project.model import ModelConfig, TwoStageLGBMRanker, load_data, make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Stage LGBM Ranker")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="outputs/model.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    cfg = ModelConfig(**raw.get("model", {}))
    cfg.data_dir = Path(raw.get("data", {}).get("data_dir", "data"))

    tx, customers, articles = load_data(cfg)
    train_full, train_feat, val_tx, _, test_gt = make_splits(tx)

    model = TwoStageLGBMRanker(cfg)
    model.fit(train_feat, val_tx, customers, articles, train_full)

    score = model.evaluate(test_gt, k=cfg.k, sample=cfg.sample_eval)
    logger.info(f"Test MAP@{cfg.k} = {score:.6f}")

    # Save only lightweight parts (no DataFrames) — same format as ensemble cache
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm_model": model.model,
        "obsolete": model._obsolete,
        "cfg": model.cfg,
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
