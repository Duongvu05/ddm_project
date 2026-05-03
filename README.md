# H&M Personalized Fashion Recommendations

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6%2B-2C7BBE?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-linter-D7FF64?logo=ruff&logoColor=black)
![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen?logo=pytest&logoColor=white)
![MAP@12](https://img.shields.io/badge/MAP%4012-0.0334-success)
![Kaggle](https://img.shields.io/badge/Kaggle-H%26M%20Fashion-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-ready two-stage recommendation system that predicts the 12 fashion articles each customer is most likely to purchase in the following week, trained on two years of H&M transaction history.

**Dataset**: [H&M Personalized Fashion Recommendations — Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)  
**Evaluation Metric**: MAP@12 (Mean Average Precision at 12) — higher is better

---

## Results

| Model | Type | MAP@12 | vs. Repurchase |
|---|---|---|---|
| Global Popularity | Baseline | 0.0029 | −88.0% |
| Age-Segmented Popularity | Baseline | 0.0035 | −85.5% |
| Recent Popularity (2w) | Baseline | 0.0068 | −71.8% |
| Item-based CF | Collaborative Filtering | 0.0086 | −64.3% |
| Repurchase | Baseline | 0.0241 | — |
| Two-Stage LGBM v1 | Main Model | 0.0280 | +16.2% |
| **Two-Stage LGBM v2** | **Main Model** | **0.0334** | **+38.6%** |
| RRF Ensemble (LGBM + Repurchase + CF) | Ensemble | 0.0298 | +23.7% |

> **Note on ensemble**: The RRF ensemble scores below the standalone LGBM v2 because repurchase and CF signals are already encoded as features inside the ranker — combining them at the output level introduces noise rather than complementary information. The LGBM v2 is the recommended production model.

---

## Architecture

```
Raw Transactions  ──►  Preprocessing & Out-of-Stock Filter
                                  │
                                  ▼
                    Stage 1 — Candidate Generation
                    (6 sources, ~200 candidates/user)
                                  │
                                  ▼
                    Stage 2 — Feature Engineering
                    (33 features: user / article / interaction)
                                  │
                                  ▼
                    Stage 3 — LightGBM LambdaRank
                    (multi-week training, NDCG@12 objective)
                                  │
                                  ▼
                         Top-12 per Customer
```

---

## Stage 1 — Candidate Generation

Each user receives up to 200 candidates sourced from multiple signals before the ranker re-orders them.

| Source | Window | Max Candidates | Signal |
|---|---|---|---|
| Repurchase (short) | Last 14 days | — | Strong recency signal |
| Repurchase (long) | Last 60 days | 50 | Habitual purchases |
| Product-code repurchase | Last 60 days | 30 | Same product, different colour/size |
| Global recent popular | Last 14 days | 100 | Trending items across all customers |
| Age-segment popular | Last 14 days | 30 | Trending within user's age cohort |
| Category popular | Last 14 days | 20 | Trending within user's favourite categories |

All candidates are filtered against a set of ~10,781 **out-of-stock articles** (≥95% of sales occurred before 2019) before being passed to Stage 2.

---

## Stage 2 — Feature Engineering (33 features)

### User features (13)
| Feature | Description |
|---|---|
| `user_age` | Customer age (float) |
| `user_is_member` | Club membership flag |
| `user_n_tx` | Total transaction count |
| `user_days_since_last_tx` | Days since most recent purchase |
| `user_avg_price` | Average spend per item |
| `user_price_std` | Price variance — captures fashion segment breadth |
| `user_n_unique_articles` | Catalogue diversity |
| `user_n_unique_categories` | Category diversity |
| `user_tx_14d` | Transactions in last 14 days (recency) |
| `user_tx_30d` | Transactions in last 30 days |
| `user_online_ratio` | Fraction of online-channel purchases |
| `user_avg_tx_per_week` | Weekly purchase cadence |
| `user_repurchase_rate` | Fraction of items bought more than once |

### Article features (10)
| Feature | Description |
|---|---|
| `art_pop_1w` | Purchase count in last 7 days |
| `art_pop_2w` | Purchase count in last 14 days |
| `art_pop_4w` | Purchase count in last 28 days |
| `art_pop_all` | All-time purchase count |
| `art_avg_price` | Average transaction price |
| `art_trend_score` | `pop_1w / (pop_4w/4 + 1)` — momentum indicator |
| `art_category_pop_2w` | Category-level popularity in last 14 days |
| `art_colour` | Encoded colour group |
| `art_product_type` | Encoded product type |
| `art_section` | Encoded section/department |

### User-Article interaction features (10)
| Feature | Description |
|---|---|
| `ua_has_purchased` | Whether the user has ever bought this article |
| `ua_purchase_count` | Number of times user bought this article |
| `ua_days_since_purchase` | Days since last purchase of this article |
| `ua_recency_score` | Exponential decay recency score |
| `ua_same_product_code` | Whether user has bought another variant of this product |
| `ua_category_purchases` | User's total purchases in this article's category |
| `ua_price_affinity` | Absolute difference between user's avg price and article price |
| `ua_candidate_source` | Which candidate generator surfaced this article |
| `ua_pop_rank_global` | Global popularity rank of this article |
| `ua_pop_rank_segment` | Popularity rank within user's age segment |

---

## Stage 3 — LightGBM LambdaRank

- **Objective**: `lambdarank` with `ndcg@12` metric
- **Multi-week training**: 3 consecutive validation weeks are used as training data (~36M feature rows), providing 8× more signal than single-week training
- **Key hyperparameters**: `n_estimators=500`, `num_leaves=127`, `learning_rate=0.05`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`
- **Label scheme**: articles purchased by the user in the target week receive label=1; all other candidates receive label=0

---

## Data Splits

The dataset covers **2018-09-20 → 2020-09-22** (~2 years). All splits are **time-based** — no random shuffling — to simulate real deployment conditions where the model never sees future data during training.

### Baseline split (`baselines.py`, `cf_model.py`)

```
◄──────────────── Train (~31M rows) ────────────────►│◄── Test (1w) ──►
                                                      │
                                              2020-09-16    2020-09-22
```

| Split | Date range | Rows | Description |
|---|---|---|---|
| Train | up to 2020-09-15 | ~31M | Full history used to fit the model |
| Test | 2020-09-16 – 2020-09-22 | ~350K | Ground truth for MAP@12 evaluation |

Used by: `GlobalPopularity`, `RecentPopularity`, `AgeSegmentedPopularity`, `RepurchaseRecommender`, `ItemCFRecommender`.

---

### LGBM split (`model.py` — `make_splits`)

The main model needs a **validation week** to build training features (features are computed from history *before* the target week, labels come *from* the target week):

```
◄──── train_feat (~30M rows) ────►│◄── val (1w) ──►│◄── test (1w) ──►
                                  │                │
                            2020-09-02        2020-09-09        2020-09-22
```

| Split | Date range | Used for |
|---|---|---|
| `train_feat` | up to 2020-09-01 | History when generating val-week candidates & features |
| `val_tx` | 2020-09-02 – 2020-09-08 | Labels for LightGBM training (purchased = 1) |
| `train_full` | up to 2020-09-08 | History when predicting on the test week |
| `test_tx` | 2020-09-09 – 2020-09-15 | Ground truth for final MAP@12 score |

> `train_full` includes the validation week so that the model's history is as recent as possible when it generates test-week candidates.

---

### Multi-week training (LGBM v2)

Instead of training on a single validation week, v2 uses **3 consecutive weeks** to create ~36M training rows:

```
Week −3  ──►  feat cutoff = 2020-08-19,  label = 2020-08-19..2020-08-26
Week −2  ──►  feat cutoff = 2020-08-26,  label = 2020-08-26..2020-09-02
Week −1  ──►  feat cutoff = 2020-09-02,  label = 2020-09-02..2020-09-09
                                                              │
                                                         (= val week)
```

For each training week the pipeline:
1. Computes candidates from history *before* `feat_cutoff`
2. Builds 33 features using that same history window
3. Assigns label=1 to articles the user actually bought in the target week
4. Appends the resulting `(X, y, group)` frame to the combined training set

All three frames are concatenated and passed to `lgb.LGBMRanker.fit()` in a single call. This multiplies the number of training examples by ~9× compared to v1 without introducing data leakage.

---

## Project Structure

```
ddm_project/
├── configs/
│   └── default.yaml              # All hyperparameters and data paths
├── data/
│   ├── articles.parquet          # Article catalogue (~105K rows)
│   ├── customers.parquet         # Customer profiles (~1.4M rows)
│   └── transactions.parquet      # 2-year transaction history (~31M rows)
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis (13 sections)
│   └── results.ipynb             # Model comparison, feature importance, error analysis
├── outputs/
│   └── lgbm_model.pkl            # Cached trained model (~2.3 MB, no DataFrames)
├── scripts/
│   ├── train.py                  # Train the Two-Stage LGBM model and save pkl
│   └── evaluate.py               # Evaluate and compare all models side-by-side
├── src/ddm_project/
│   ├── preprocessing.py          # Memory optimisation, out-of-stock filter, week splits
│   ├── baselines.py              # Four baseline recommenders + MAP@K metric
│   ├── cf_model.py               # Item-based Collaborative Filtering
│   ├── model.py                  # Two-Stage LightGBM Ranker (main model)
│   └── ensemble.py               # RRF Ensemble + LGBM cache helpers
├── tests/
│   ├── test_metrics.py           # MAP@K unit tests (8 cases)
│   └── test_preprocessing.py     # Preprocessing utility tests (4 cases)
├── ecosystem.config.js           # PM2 process definitions for remote training
└── pyproject.toml                # Dependencies, tool config (ruff, pyright, pytest)
```

---

## Setup & Installation

**Requirements**: Python ≥ 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone <repo-url>
cd ddm_project

# Install all dependencies
uv sync

# Verify installation
uv run pytest
```

---

## Running the Project

### 1. Verify the installation

```bash
uv run pytest
```

All 12 tests should pass. This confirms that the data utilities and metric functions work correctly before loading the full dataset.

---

### 2. Run the baselines

```bash
uv run python -m ddm_project.baselines
```

Fits and evaluates all four baselines in sequence and prints a MAP@12 comparison table. Expected runtime: **~2 minutes** (no heavy feature engineering).

```
GlobalPopularityRecommender      MAP@12 = 0.002938
AgeSegmentedPopularity           MAP@12 = 0.003541
RecentPopularityRecommender      MAP@12 = 0.006824
RepurchaseRecommender            MAP@12 = 0.024112
```

---

### 3. Run Item-based CF

```bash
uv run python -m ddm_project.cf_model
```

Builds a sparse co-purchase similarity matrix and evaluates on the test week. Expected runtime: **~5 minutes**.

```
ItemCFRecommender                MAP@12 = 0.008621
```

---

### 4. Train the main LGBM model

```bash
uv run python scripts/train.py --config configs/default.yaml --output outputs/lgbm_model.pkl
```

What happens step-by-step:
1. Loads `data/*.parquet` and applies memory optimisation
2. Runs `make_splits` to produce `train_feat`, `val_tx`, `train_full`, `test_gt`
3. Generates candidates for 3 training weeks and builds 33 features (~36M rows)
4. Fits `LGBMRanker` with `lambdarank` / `ndcg@12` objective
5. Evaluates on `test_gt` and logs `MAP@12`
6. Saves a lightweight pickle to `outputs/lgbm_model.pkl` (~2.3 MB, **no DataFrames**)

Expected runtime: **~20–30 minutes** on a 16-core CPU.

```
Test MAP@12 = 0.033412
Model saved → outputs/lgbm_model.pkl  (2.3 MB)
```

To use a custom config:

```bash
uv run python scripts/train.py --config configs/my_experiment.yaml --output outputs/my_model.pkl
```

---

### 5. Evaluate all models side-by-side

```bash
uv run python scripts/evaluate.py --config configs/default.yaml
```

Runs every model (baselines, CF, LGBM) and prints a ranked comparison table. Loads `outputs/lgbm_model.pkl` if it exists to avoid retraining. Expected runtime: **~10 minutes** (LGBM evaluation only; no refitting if cache exists).

---

### 6. Run the RRF Ensemble

```bash
uv run python -m ddm_project.ensemble
```

- Loads the cached LGBM model from `outputs/lgbm_model.pkl` (retrains if missing)
- Fits `RepurchaseRecommender` and `ItemCFRecommender` on `train_full`
- Fuses the three models' ranked lists with Reciprocal Rank Fusion
- Prints MAP@12 for each individual model and the ensemble

Expected runtime: **~15 minutes** (cache hit) or **~45 minutes** (cache miss).

---

### 7. Step-by-step reproduction of all reported results

```bash
# 1. Install
uv sync

# 2. Sanity check
uv run pytest

# 3. Baselines
uv run python -m ddm_project.baselines

# 4. CF
uv run python -m ddm_project.cf_model

# 5. Train LGBM (saves cache)
uv run python scripts/train.py --config configs/default.yaml --output outputs/lgbm_model.pkl

# 6. Full comparison table
uv run python scripts/evaluate.py --config configs/default.yaml

# 7. Ensemble
uv run python -m ddm_project.ensemble
```

---

## Remote Training with PM2

For long training jobs on a remote server, use the included `ecosystem.config.js`.

### One-time setup

```bash
# If Node.js is available
npm i -g pm2

# Without sudo (via NVM)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts
npm i -g pm2
```

### Available PM2 jobs

| App name | Command | Description |
|---|---|---|
| `hm-train` | `scripts/train.py` | Train LGBM and save pkl |
| `hm-ensemble` | `ddm_project.ensemble` | Run full ensemble evaluation |
| `hm-evaluate` | `scripts/evaluate.py` | Compare all models |

### Workflow

```bash
# Step 1 — Start a job (safe: autorestart is disabled)
pm2 start ecosystem.config.js --only hm-train

# Step 2 — Monitor (can safely disconnect SSH after this)
pm2 logs hm-train --lines 50

# Step 3 — Check status
pm2 ls

# Step 4 — Stop / clean up
pm2 stop hm-train
pm2 delete hm-train
```

> `autorestart: false` and `max_restarts: 0` are set in `ecosystem.config.js` to prevent the job from looping if it exits normally or crashes.

---

## EDA Coverage (`notebooks/eda.ipynb`)

| Section | Topic |
|---|---|
| 1–2 | Schema overview, missing values, data types |
| 3–4 | Weekly activity trends, price distribution |
| 5–6 | Customer demographics, purchase frequency distribution |
| 7–8 | Article catalogue analysis, top popular items |
| 9 | Age group × product group purchase heatmap |
| 10 | **Seasonality** — monthly price trends, PCA + GMM product clustering |
| 11 | **Out-of-stock detection** — ~10,781 discontinued articles identified |
| 12 | **Repurchase rate** at three granularities (article / product code / category) |
| 13 | Word cloud of article names and descriptions |

---

## Key Findings

1. **Repurchase is the strongest single signal.** H&M customers frequently rebuy the same items; the repurchase baseline (MAP@12 = 0.0241) is difficult to beat with popularity-only models and serves as the main comparison point.

2. **Recency dominates history depth.** Recent popularity (last 2 weeks, 0.0068) outperforms all-time global popularity (0.0029) by over 2×, confirming that fashion demand is highly seasonal and short-lived.

3. **~10,781 articles are effectively out of stock.** Items where ≥95% of sales occurred before 2019 are excluded from all candidate pools, removing noise and improving precision.

4. **Candidate recall was the primary bottleneck.** Expanding from 2 candidate sources to 6 and increasing the candidate budget from 50 to 200 raised Stage 1 recall from ~9.6% to ~28%, which directly drove the v2 MAP@12 improvement.

5. **Multi-week training provides large gains.** Using 3 consecutive training windows (~36M rows vs. ~4M) allows the ranker to observe more user-article pairs and generalise better across purchase patterns.

6. **Cold-start remains the dominant failure mode.** Approximately 60% of test users receive an AP@12 of 0, driven by customers with sparse or no purchase history who cannot benefit from repurchase or interaction features.

7. **Top LightGBM features by gain**: `ua_days_since_purchase`, `ua_has_purchased`, `art_pop_2w`, `user_days_since_last_tx`, `ua_purchase_count`, `art_trend_score`.

8. **Ensemble added noise, not signal.** Because repurchase and CF signals are already encoded as explicit features inside the LGBM ranker, fusing their output-level lists via RRF hurts precision. A more effective ensemble strategy would combine multiple LGBM variants trained with different hyperparameters or candidate sets.

---

## Configuration (`configs/default.yaml`)

All model hyperparameters, data paths, and evaluation settings are centralised in `configs/default.yaml`. The training script and evaluate script both accept `--config` to point to an alternative config file, enabling reproducible experiments.

---

## Reproducibility

All random seeds are fixed (`seed=42` throughout). Results may vary slightly across platforms due to floating-point differences in LightGBM's parallel tree construction, but MAP@12 should remain within ±0.001 of the reported values.
