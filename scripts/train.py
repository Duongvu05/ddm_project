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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

    # --- BƯỚC 1: LƯU FILE PICKLE TRƯỚC (Đảm bảo an toàn dữ liệu) ---
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm_model": model.model,
        "obsolete": model._obsolete,
        "cfg": model.cfg,
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved successfully → {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # --- BƯỚC 2: TRÍCH XUẤT FEATURE IMPORTANCE (Sửa lỗi AttributeError) ---
    try:
        logger.info("Extracting feature importance...")
        lgbm_obj = model.model
        
        # Kiểm tra xem là LGBMRanker (Sklearn API) hay Booster thuần
        if hasattr(lgbm_obj, "feature_name_"):
            # Đối với LGBMRanker
            feats = lgbm_obj.feature_name_
            imps = lgbm_obj.feature_importances_
        else:
            # Đối với Booster thuần
            feats = lgbm_obj.feature_name()
            imps = lgbm_obj.feature_importance(importance_type='gain')

        importance_df = pd.DataFrame({
            'feature': feats,
            'importance': imps
        }).sort_values(by='importance', ascending=False)

        # Tạo thư mục figures
        fig_dir = Path("figures")
        fig_dir.mkdir(exist_ok=True)

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature', palette='viridis')
        plt.title(f"Top 20 Feature Importance - MAP: {score:.4f}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()

        # Lưu ảnh
        fig_path = fig_dir / f"feature_importance_{out.stem}.png"
        plt.savefig(fig_path)
        logger.info(f"Feature importance plot saved → {fig_path}")

    except Exception as e:
        # Nếu có lỗi khi vẽ ảnh, logger sẽ báo nhưng model đã được lưu an toàn ở Bước 1
        logger.error(f"Failed to generate feature importance plot: {e}")

if __name__ == "__main__":
    main()
