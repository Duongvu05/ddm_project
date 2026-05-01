"""Two-stage recommendation model: Candidate Generation → LightGBM Re-ranker.

Improvements over v1
--------------------
Candidate Generation:
  + Product-code repurchase  — user bought a different variant of the same product
  + Category-popular         — top items in categories the user actively shops
  + Out-of-stock filtering   — remove articles with ≥95% sales before 2019
  + n_candidates 100 → 200

Feature Engineering (+5 new features, 33 total):
  + art_trend_score          — pop_1w / (pop_4w/4 + 1): trending up or down?
  + art_category_pop_2w      — how popular is this product group in the last 2 weeks?
  + ua_same_product_code     — did the user buy a different variant of this exact product?
  + ua_category_purchases    — how many times has the user bought from this product group?
  + ua_price_affinity        — |article_price − user_avg_price| / user_avg_price
  + user_price_std           — variability in user's spending
  + user_online_ratio        — fraction of purchases made online (channel 2)

Training:
  + Multi-week training      — 3 validation weeks instead of 1 (~3× more training data)

LightGBM:
  + n_estimators 300 → 500
  + min_child_samples 20 → 10

Timeline (unchanged):
  test_start = 2020-09-16 | val_start = 2020-09-09 | training weeks = 3, 4, 5 before test

Metric: MAP@12
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from ddm_project.baselines import map_at_k
from ddm_project.preprocessing import find_obsolete_articles


# ── Config ────────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    data_dir: Path = Path("data")
    k: int = 12
    n_candidates: int = 200
    repurchase_short_days: int = 14
    repurchase_long_days: int = 60
    popular_global_n: int = 100
    popular_segment_n: int = 50
    popular_window_days: int = 14
    category_popular_n: int = 20       # top-N per user category
    n_train_weeks: int = 3             # number of validation weeks to train on
    age_bins: list[int] = [15, 25, 35, 45, 55, 65, 100]
    age_labels: list[str] = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    lgbm_params: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [12],
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 500,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }
    sample_eval: int = 50_000

    @field_validator("k")
    @classmethod
    def k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be positive")
        return v


# ── Data loading ──────────────────────────────────────────────────────────────


def load_data(cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading data …")
    tx = pd.read_parquet(cfg.data_dir / "transactions.parquet")
    customers = pd.read_parquet(cfg.data_dir / "customers.parquet")
    articles = pd.read_parquet(cfg.data_dir / "articles.parquet")
    logger.info(f"transactions={len(tx):,}  customers={len(customers):,}  articles={len(articles):,}")
    return tx, customers, articles


def make_splits(
    tx: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    """Return (train_full, train_feat, val_tx, val_gt, test_gt)."""
    last = tx["t_dat"].max()
    test_start = last - pd.Timedelta(days=6)
    val_start = test_start - pd.Timedelta(days=7)

    train_feat = tx[tx["t_dat"] < val_start].copy()
    val_tx = tx[(tx["t_dat"] >= val_start) & (tx["t_dat"] < test_start)].copy()
    train_full = tx[tx["t_dat"] < test_start].copy()
    test_tx = tx[tx["t_dat"] >= test_start].copy()

    val_gt: dict[str, set[str]] = val_tx.groupby("customer_id")["article_id"].apply(set).to_dict()
    test_gt: dict[str, set[str]] = test_tx.groupby("customer_id")["article_id"].apply(set).to_dict()

    logger.info(
        f"train_feat={len(train_feat):,}  val_tx={len(val_tx):,}  "
        f"val_users={len(val_gt):,}  test_users={len(test_gt):,}"
    )
    return train_full, train_feat, val_tx, val_gt, test_gt


# ── Stage 1: Candidate Generation ────────────────────────────────────────────


def _age_group_col(customers: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    customers = customers.copy()
    customers["age_group"] = pd.cut(
        customers["age"], bins=cfg.age_bins, labels=cfg.age_labels, right=True
    ).astype(str)
    customers.loc[customers["age"].isna(), "age_group"] = "unknown"
    return customers


def generate_candidates(
    history: pd.DataFrame,
    target_users: list[str],
    customers: pd.DataFrame,
    cfg: ModelConfig,
    articles: pd.DataFrame | None = None,
    obsolete_articles: set[str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns (customer_id, article_id, source)."""
    t_max = history["t_dat"].max()
    short_cutoff = t_max - pd.Timedelta(days=cfg.repurchase_short_days)
    long_cutoff = t_max - pd.Timedelta(days=cfg.repurchase_long_days)
    pop_cutoff = t_max - pd.Timedelta(days=cfg.popular_window_days)

    target_set = set(target_users)
    history_target = history[history["customer_id"].isin(target_set)]
    recent = history[history["t_dat"] >= pop_cutoff]

    # Pre-compute obsolete set if not provided
    if obsolete_articles is None:
        obsolete_articles = set()

    candidates: list[pd.DataFrame] = []

    # ── Source 1: repurchase short window ────────────────────────────────────
    short_hist = history_target[history_target["t_dat"] >= short_cutoff]
    if len(short_hist):
        c = short_hist[["customer_id", "article_id"]].drop_duplicates().copy()
        c["source"] = "repurchase_short"
        candidates.append(c)

    # ── Source 2: repurchase long window ─────────────────────────────────────
    long_hist = history_target[history_target["t_dat"] >= long_cutoff]
    if len(long_hist):
        c = long_hist[["customer_id", "article_id"]].drop_duplicates().copy()
        c["source"] = "repurchase_long"
        candidates.append(c)

    # ── Source 3: product-code repurchase (NEW) ───────────────────────────────
    # User bought a different colour/size of the same product → recommend popular
    # variants of product codes the user has purchased before.
    if articles is not None:
        art_pc = articles.set_index("article_id")["product_code"].to_dict()
        # product-code → popular articles (recent)
        recent_pc = recent.copy()
        recent_pc["product_code"] = recent_pc["article_id"].map(art_pc)
        pc_popular = (
            recent_pc.groupby(["product_code", "article_id"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .groupby("product_code")
            .head(3)   # top-3 variants per product code
            .set_index("product_code")["article_id"]
        )
        pc_popular_dict: dict[str, list[str]] = (
            pc_popular.groupby(level=0).apply(list).to_dict()
        )
        # user → product codes bought
        user_pcs = (
            history_target.assign(pc=history_target["article_id"].map(art_pc))
            .groupby("customer_id")["pc"]
            .apply(set)
            .to_dict()
        )
        pc_rows: list[dict[str, str]] = []
        for uid in target_users:
            for pc in user_pcs.get(uid, set()):
                for art in pc_popular_dict.get(pc, []):
                    pc_rows.append({"customer_id": uid, "article_id": art,
                                    "source": "product_code_repurchase"})
        if pc_rows:
            candidates.append(pd.DataFrame(pc_rows))

    # ── Source 4: global recent popular ──────────────────────────────────────
    global_top = (
        recent["article_id"].value_counts().head(cfg.popular_global_n).index.tolist()
    )
    rows = pd.DataFrame({"customer_id": np.repeat(target_users, len(global_top)),
                         "article_id": global_top * len(target_users),
                         "source": "popular_global"})
    candidates.append(rows)

    # ── Source 5: age-segment recent popular ─────────────────────────────────
    customers_ag = _age_group_col(customers, cfg)
    tx_age = recent.merge(
        customers_ag[["customer_id", "age_group"]], on="customer_id", how="left"
    )
    segment_tops: dict[str, list[str]] = {}
    for grp, grp_df in tx_age.groupby("age_group", observed=True):
        segment_tops[str(grp)] = (
            grp_df["article_id"].value_counts().head(cfg.popular_segment_n).index.tolist()
        )
    cust_seg = customers_ag.set_index("customer_id")["age_group"].to_dict()
    seg_rows: list[dict[str, str]] = []
    for uid in target_users:
        seg = cust_seg.get(uid, "unknown")
        for art in segment_tops.get(seg, []):
            seg_rows.append({"customer_id": uid, "article_id": art, "source": "popular_segment"})
    if seg_rows:
        candidates.append(pd.DataFrame(seg_rows))

    # ── Source 6: category-popular (NEW) ─────────────────────────────────────
    # For each user, find their top-3 most purchased product groups, then
    # recommend the most popular articles in those categories recently.
    if articles is not None:
        art_pg = articles.set_index("article_id")["product_group_name"].to_dict()
        recent_pg = recent.copy()
        recent_pg["product_group"] = recent_pg["article_id"].map(art_pg)
        pg_popular: dict[str, list[str]] = (
            recent_pg.groupby(["product_group", "article_id"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .groupby("product_group")
            .head(cfg.category_popular_n)
            .groupby("product_group")["article_id"]
            .apply(list)
            .to_dict()
        )
        user_pg_counts = (
            history_target.assign(pg=history_target["article_id"].map(art_pg))
            .groupby(["customer_id", "pg"])
            .size()
            .reset_index(name="cnt")
            .sort_values("cnt", ascending=False)
            .groupby("customer_id")
            .head(3)   # top-3 product groups per user
        )
        cat_rows: list[dict[str, str]] = []
        for _, row in user_pg_counts.iterrows():
            uid, pg = row["customer_id"], row["pg"]
            for art in pg_popular.get(pg, []):
                cat_rows.append({"customer_id": uid, "article_id": art,
                                 "source": "category_popular"})
        if cat_rows:
            candidates.append(pd.DataFrame(cat_rows))

    # ── Merge, filter obsolete, trim ─────────────────────────────────────────
    all_cands = pd.concat(candidates, ignore_index=True)
    if obsolete_articles:
        all_cands = all_cands[~all_cands["article_id"].isin(obsolete_articles)]

    # keep first occurrence (repurchase > popularity)
    all_cands = all_cands.drop_duplicates(subset=["customer_id", "article_id"])
    all_cands = (
        all_cands.groupby("customer_id", group_keys=False)
        .head(cfg.n_candidates)
        .reset_index(drop=True)
    )

    n_users = all_cands["customer_id"].nunique()
    logger.info(
        f"Candidates: {len(all_cands):,} rows for {n_users:,} users"
        f"  (avg {len(all_cands)/n_users:.1f} per user)"
    )
    return all_cands


# ── Stage 2: Feature Engineering ─────────────────────────────────────────────


def build_user_features(history: pd.DataFrame, customers: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    t_max = history["t_dat"].max()

    user_agg = history.groupby("customer_id").agg(
        user_total_tx=("article_id", "count"),
        user_unique_articles=("article_id", "nunique"),
        user_avg_price=("price", "mean"),
        user_price_std=("price", "std"),      # NEW: price variability
        user_last_tx_date=("t_dat", "max"),
    ).reset_index()
    user_agg["user_days_since_last_tx"] = (t_max - user_agg["user_last_tx_date"]).dt.days
    user_agg["user_price_std"] = user_agg["user_price_std"].fillna(0)
    user_agg = user_agg.drop(columns=["user_last_tx_date"])

    # Recent activity (2 weeks)
    cutoff_2w = t_max - pd.Timedelta(weeks=2)
    recent_agg = (
        history[history["t_dat"] >= cutoff_2w]
        .groupby("customer_id")
        .agg(user_tx_2w=("article_id", "count"))
        .reset_index()
    )
    user_agg = user_agg.merge(recent_agg, on="customer_id", how="left")
    user_agg["user_tx_2w"] = user_agg["user_tx_2w"].fillna(0).astype(int)

    # Online ratio (NEW): fraction of purchases via channel 2
    if "sales_channel_id" in history.columns:
        online = (
            history.groupby("customer_id")
            .apply(lambda g: (g["sales_channel_id"] == 2).mean(), include_groups=False)
            .rename("user_online_ratio")
            .reset_index()
        )
        user_agg = user_agg.merge(online, on="customer_id", how="left")
    else:
        user_agg["user_online_ratio"] = 0.5

    # Demographics
    cust = _age_group_col(customers, cfg)[
        ["customer_id", "age", "age_group", "FN", "Active", "club_member_status", "fashion_news_frequency"]
    ].copy()
    cust["club_active"] = (cust["club_member_status"] == "ACTIVE").astype(int)
    cust["news_regular"] = (cust["fashion_news_frequency"] == "Regularly").astype(int)
    cust["FN"] = cust["FN"].fillna(0).astype(int)
    cust["Active"] = cust["Active"].fillna(0).astype(int)
    age_order = {v: i for i, v in enumerate(cfg.age_labels + ["unknown"])}
    cust["age_group_enc"] = cust["age_group"].map(age_order).fillna(len(age_order))
    cust = cust.drop(columns=["club_member_status", "fashion_news_frequency", "age_group"])

    user_feats = user_agg.merge(cust, on="customer_id", how="left")
    user_feats["age"] = user_feats["age"].fillna(user_feats["age"].median())
    return user_feats


def build_article_features(
    history: pd.DataFrame, articles: pd.DataFrame
) -> pd.DataFrame:
    t_max = history["t_dat"].max()

    def pop_window(days: int, col: str) -> pd.Series:
        return (
            history[history["t_dat"] >= t_max - pd.Timedelta(days=days)]
            .groupby("article_id")["customer_id"]
            .count()
            .rename(col)
        )

    pop_all = history.groupby("article_id").agg(
        art_total_tx=("customer_id", "count"),
        art_unique_customers=("customer_id", "nunique"),
        art_avg_price=("price", "mean"),
        art_last_sale=("t_dat", "max"),
    ).reset_index()

    for days, col in [(7, "art_pop_1w"), (14, "art_pop_2w"), (28, "art_pop_4w")]:
        art_feats_tmp = pop_window(days, col)
        pop_all = pop_all.merge(art_feats_tmp, on="article_id", how="left")

    for c in ["art_pop_1w", "art_pop_2w", "art_pop_4w"]:
        pop_all[c] = pop_all[c].fillna(0).astype(int)

    pop_all["art_days_since_last_sale"] = (t_max - pop_all["art_last_sale"]).dt.days
    # Trend score (NEW): is the article gaining momentum?
    pop_all["art_trend_score"] = pop_all["art_pop_1w"] / (pop_all["art_pop_4w"] / 4 + 1)
    pop_all = pop_all.drop(columns=["art_last_sale"])

    # Category-level popularity in last 2 weeks (NEW)
    art_pg = articles.set_index("article_id")["product_group_name"].to_dict()
    recent_2w = history[history["t_dat"] >= t_max - pd.Timedelta(weeks=2)].copy()
    recent_2w["product_group"] = recent_2w["article_id"].map(art_pg)
    cat_pop = (
        recent_2w.groupby("product_group")
        .size()
        .rename("art_category_pop_2w")
        .reset_index()
    )
    articles_with_cat = articles[["article_id", "product_group_name"]].merge(
        cat_pop, left_on="product_group_name", right_on="product_group", how="left"
    )[["article_id", "art_category_pop_2w"]]
    pop_all = pop_all.merge(articles_with_cat, on="article_id", how="left")
    pop_all["art_category_pop_2w"] = pop_all["art_category_pop_2w"].fillna(0).astype(int)

    # Article metadata
    meta = articles[[
        "article_id", "product_code", "product_type_no", "product_group_name",
        "graphical_appearance_no", "colour_group_code",
        "index_group_no", "section_no", "garment_group_no",
    ]].copy()
    meta["product_group_enc"] = meta["product_group_name"].astype("category").cat.codes
    meta = meta.drop(columns=["product_group_name"])

    art_feats = pop_all.merge(meta, on="article_id", how="left")
    return art_feats


def build_user_article_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    t_max = history["t_dat"].max()

    # Per (user, article): purchase count + recency
    ua = (
        history.groupby(["customer_id", "article_id"])
        .agg(ua_purchase_count=("t_dat", "count"), ua_last_purchase=("t_dat", "max"))
        .reset_index()
    )
    ua["ua_days_since_purchase"] = (t_max - ua["ua_last_purchase"]).dt.days
    ua = ua.drop(columns=["ua_last_purchase"])

    merged = candidates.merge(ua, on=["customer_id", "article_id"], how="left")
    merged["ua_has_purchased"] = merged["ua_purchase_count"].notna().astype(int)
    merged["ua_purchase_count"] = merged["ua_purchase_count"].fillna(0).astype(int)
    merged["ua_days_since_purchase"] = merged["ua_days_since_purchase"].fillna(9999).astype(int)

    # Same product_code purchased (NEW)
    art_pc = articles.set_index("article_id")["product_code"].to_dict()
    user_pcs = (
        history.groupby("customer_id")["article_id"]
        .apply(lambda s: set(art_pc.get(a, "") for a in s))
        .to_dict()
    )
    merged["ua_same_product_code"] = merged.apply(
        lambda r: int(art_pc.get(r["article_id"], "__") in user_pcs.get(r["customer_id"], set())),
        axis=1,
    )

    # Category purchases by user (NEW)
    art_pg = articles.set_index("article_id")["product_group_name"].to_dict()
    user_cat_counts = (
        history.assign(pg=history["article_id"].map(art_pg))
        .groupby(["customer_id", "pg"])
        .size()
        .reset_index(name="ua_category_purchases")
    )
    merged = merged.assign(pg=merged["article_id"].map(art_pg))
    merged = merged.merge(user_cat_counts, on=["customer_id", "pg"], how="left")
    merged["ua_category_purchases"] = merged["ua_category_purchases"].fillna(0).astype(int)
    merged = merged.drop(columns=["pg"])

    # Source priority
    source_priority = {
        "repurchase_short": 6,
        "repurchase_long": 5,
        "product_code_repurchase": 4,
        "category_popular": 3,
        "popular_segment": 2,
        "popular_global": 1,
    }
    merged["candidate_source_enc"] = merged["source"].map(source_priority).fillna(0).astype(int)
    return merged


def build_features(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    cfg: ModelConfig,
) -> pd.DataFrame:
    logger.info("Building user features …")
    user_feats = build_user_features(history, customers, cfg)

    logger.info("Building article features …")
    art_feats = build_article_features(history, articles)

    logger.info("Building user-article features …")
    df = build_user_article_features(history, candidates, articles)

    df = df.merge(user_feats, on="customer_id", how="left")
    df = df.merge(art_feats, on="article_id", how="left")

    # Price affinity (NEW): normalised distance between article price and user avg price
    df["ua_price_affinity"] = (
        (df["art_avg_price"] - df["user_avg_price"]).abs()
        / (df["user_avg_price"].clip(lower=1e-6))
    ).fillna(0).astype(np.float32)

    return df


FEATURE_COLS = [
    # user
    "user_total_tx", "user_unique_articles", "user_avg_price", "user_price_std",
    "user_days_since_last_tx", "user_tx_2w", "user_online_ratio",
    "age", "age_group_enc", "FN", "Active", "club_active", "news_regular",
    # article
    "art_total_tx", "art_unique_customers", "art_avg_price",
    "art_days_since_last_sale", "art_trend_score", "art_category_pop_2w",
    "art_pop_1w", "art_pop_2w", "art_pop_4w",
    "product_type_no", "product_group_enc", "graphical_appearance_no",
    "colour_group_code", "index_group_no", "section_no", "garment_group_no",
    # user-article
    "ua_has_purchased", "ua_purchase_count", "ua_days_since_purchase",
    "ua_same_product_code", "ua_category_purchases", "ua_price_affinity",
    "candidate_source_enc",
]


# ── Stage 3: LightGBM Ranker ──────────────────────────────────────────────────


class TwoStageLGBMRanker:
    """Candidate generation + LightGBM LambdaRank re-ranker."""

    name = "two_stage_lgbm_v2"

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self.model: lgb.LGBMRanker | None = None
        self._customers: pd.DataFrame | None = None
        self._articles: pd.DataFrame | None = None
        self._train_full: pd.DataFrame | None = None
        self._obsolete: set[str] = set()

    def _make_training_frame(
        self,
        history: pd.DataFrame,
        customers: pd.DataFrame,
        articles: pd.DataFrame,
        label_tx: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        target_users = label_tx["customer_id"].unique().tolist()
        purchased = set(zip(label_tx["customer_id"], label_tx["article_id"]))

        cands = generate_candidates(
            history, target_users, customers, self.cfg,
            articles=articles, obsolete_articles=self._obsolete,
        )
        df = build_features(history, cands, customers, articles, self.cfg)
        df = df.dropna(subset=FEATURE_COLS)
        df["label"] = df.apply(
            lambda r: 1 if (r["customer_id"], r["article_id"]) in purchased else 0,
            axis=1,
        )
        df = df.sort_values("customer_id").reset_index(drop=True)
        group_sizes = df.groupby("customer_id", sort=False).size().values
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df["label"].values.astype(np.int32)
        return df, X, y, group_sizes

    def fit(
        self,
        train_feat: pd.DataFrame,
        val_tx: pd.DataFrame,
        customers: pd.DataFrame,
        articles: pd.DataFrame,
        train_full: pd.DataFrame,
    ) -> None:
        self._customers = customers
        self._articles = articles
        self._train_full = train_full
        self._obsolete = find_obsolete_articles(train_full)

        # Multi-week training: collect training frames from n_train_weeks weeks
        logger.info(f"─── Building training data ({self.cfg.n_train_weeks} weeks) ───")
        t0 = time.time()

        last = train_full["t_dat"].max()
        test_start = last - pd.Timedelta(days=6)
        # week offsets relative to test_start: val week = 1, then 2, 3 ...
        all_X, all_y, all_groups = [], [], []

        for week_idx in range(1, self.cfg.n_train_weeks + 1):
            label_end = test_start - pd.Timedelta(weeks=week_idx - 1)
            label_start = label_end - pd.Timedelta(weeks=1)
            feat_cutoff = label_start

            hist_w = train_full[train_full["t_dat"] < feat_cutoff].copy()
            label_w = train_full[
                (train_full["t_dat"] >= label_start) & (train_full["t_dat"] < label_end)
            ].copy()

            if len(label_w) == 0 or len(hist_w) == 0:
                continue

            logger.info(
                f"  Week -{week_idx}: feat_cutoff={feat_cutoff.date()}  "
                f"label={label_start.date()}..{label_end.date()}  "
                f"label_rows={len(label_w):,}"
            )
            _, X_w, y_w, g_w = self._make_training_frame(hist_w, customers, articles, label_w)
            all_X.append(X_w)
            all_y.append(y_w)
            all_groups.append(g_w)

        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        groups_train = np.concatenate(all_groups)

        pos = y_train.sum()
        logger.info(
            f"Combined training frame: {len(X_train):,} rows  "
            f"pos={pos:,}  neg={len(y_train)-pos:,}  ({time.time() - t0:.1f}s)"
        )

        logger.info("─── Training LightGBM Ranker ───")
        self.model = lgb.LGBMRanker(**self.cfg.lgbm_params)
        self.model.fit(
            X_train, y_train,
            group=groups_train,
            feature_name=FEATURE_COLS,
            callbacks=[lgb.log_evaluation(period=100)],
        )
        logger.info("Training complete.")

    def predict(self, customer_ids: list[str], k: int = 12) -> dict[str, list[str]]:
        assert self.model is not None, "Call fit() first"

        cands = generate_candidates(
            self._train_full, customer_ids, self._customers, self.cfg,
            articles=self._articles, obsolete_articles=self._obsolete,
        )
        df = build_features(
            self._train_full, cands, self._customers, self._articles, self.cfg
        )
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0.0

        X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        df = df.copy()
        df["score"] = self.model.predict(X)

        preds: dict[str, list[str]] = {}
        for uid, grp in df.groupby("customer_id"):
            preds[str(uid)] = grp.nlargest(k, "score")["article_id"].tolist()
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

        logger.info(f"Predicting for {len(gt_sub):,} users …")
        preds = self.predict(list(gt_sub.keys()), k=k)
        score = map_at_k(preds, gt_sub, k=k)
        logger.info(f"[two_stage_lgbm_v2] MAP@{k} = {score:.6f}")
        return score


# ── Runner ────────────────────────────────────────────────────────────────────


def run(cfg: ModelConfig | None = None) -> dict[str, float]:
    if cfg is None:
        cfg = ModelConfig()

    tx, customers, articles = load_data(cfg)
    train_full, train_feat, val_tx, _val_gt, test_gt = make_splits(tx)

    model = TwoStageLGBMRanker(cfg)
    model.fit(train_feat, val_tx, customers, articles, train_full)

    logger.info("─── Evaluating on test week ───")
    score = model.evaluate(test_gt, k=cfg.k, sample=cfg.sample_eval)

    from ddm_project.baselines import (
        RepurchaseRecommender,
        RecentPopularityRecommender,
        train_test_split,
    )
    _, _, gt_b = train_test_split(tx)
    results = {"two_stage_lgbm_v2": score}
    for b in [RepurchaseRecommender(), RecentPopularityRecommender()]:
        b.fit(train_full)
        results[b.name] = b.evaluate(gt_b, k=cfg.k, sample=cfg.sample_eval)

    logger.info("─── Final Results ───")
    for name, s in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◄" if name == "two_stage_lgbm_v2" else ""
        logger.info(f"  {name:<40} MAP@{cfg.k} = {s:.6f}{marker}")

    return results


if __name__ == "__main__":
    run()
