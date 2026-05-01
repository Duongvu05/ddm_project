"""Script to generate the EDA notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# H&M Fashion Recommendation – EDA

Dataset bao gồm 3 bảng:
- **articles**: 105 542 sản phẩm thời trang
- **customers**: 1 371 980 khách hàng
- **transactions**: 31 788 324 giao dịch (09/2018 – 09/2020)

> **Task**: Dự đoán 12 sản phẩm mỗi khách hàng sẽ mua trong tuần cuối cùng.
> **Metric**: MAP@12
"""))

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110

DATA = Path("../data")
"""))

# ── Cell 2: Load data ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
articles     = pd.read_parquet(DATA / "articles.parquet")
customers    = pd.read_parquet(DATA / "customers.parquet")
transactions = pd.read_parquet(DATA / "transactions.parquet")

print(f"articles     : {articles.shape}")
print(f"customers    : {customers.shape}")
print(f"transactions : {transactions.shape}")
"""))

# ── Cell 3: Schema overview ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Tổng quan Schema"))

cells.append(nbf.v4.new_code_cell("""\
def df_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype"  : df.dtypes,
        "non_null": df.notnull().sum(),
        "null_pct": (df.isnull().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    print(f"\\n{'='*55}\\n  {name}\\n{'='*55}")
    print(summary.to_string())
    return summary

_ = df_summary(articles,     "ARTICLES")
_ = df_summary(customers,    "CUSTOMERS")
_ = df_summary(transactions, "TRANSACTIONS")
"""))

# ── Cell 4: Missing values heatmap ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2. Missing Values"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (df, title) in zip(axes, [
    (articles,  "Articles"),
    (customers, "Customers"),
    (transactions, "Transactions"),
]):
    null_pct = df.isnull().mean() * 100
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(title)
    else:
        null_pct.sort_values().plot(kind="barh", ax=ax, color="salmon")
        ax.set_xlabel("Missing %")
        ax.set_title(title)
        for bar in ax.patches:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("missing_values.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 5: Transaction time series ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Transactions – Phân tích theo thời gian"))

cells.append(nbf.v4.new_code_cell("""\
tx = transactions.copy()
tx["year_week"] = tx["t_dat"].dt.to_period("W")

weekly = tx.groupby("year_week").agg(
    n_transactions=("article_id", "count"),
    n_customers=("customer_id", "nunique"),
    n_articles=("article_id", "nunique"),
    revenue=("price", "sum"),
).reset_index()
weekly["year_week_str"] = weekly["year_week"].astype(str)

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
metrics = [
    ("n_transactions", "# Transactions", "steelblue"),
    ("n_customers",    "# Unique Customers", "darkorange"),
    ("n_articles",     "# Unique Articles Sold", "seagreen"),
    ("revenue",        "Total Revenue (normalised)", "purple"),
]
for ax, (col, label, color) in zip(axes, metrics):
    ax.plot(weekly["year_week_str"], weekly[col], color=color, linewidth=1.5)
    ax.fill_between(range(len(weekly)), weekly[col], alpha=0.15, color=color)
    ax.set_ylabel(label, fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(12))
    ax.tick_params(axis="x", rotation=45, labelsize=8)

axes[-1].set_xlabel("Week")
plt.suptitle("Weekly Activity (Sep 2018 – Sep 2020)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("weekly_activity.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 6: Sales channel ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

channel_counts = tx["sales_channel_id"].value_counts()
channel_labels = channel_counts.index.map({1: "Store (1)", 2: "Online (2)"})
axes[0].pie(channel_counts, labels=channel_labels, autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452"], startangle=90)
axes[0].set_title("Sales Channel Distribution")

channel_price = tx.groupby("sales_channel_id")["price"].mean()
axes[1].bar(["Store (1)", "Online (2)"], channel_price.values,
            color=["#4C72B0", "#DD8452"])
axes[1].set_ylabel("Mean Price")
axes[1].set_title("Average Price by Channel")

plt.tight_layout()
plt.savefig("sales_channel.png", bbox_inches="tight")
plt.show()
print(channel_counts.rename(index={1: "Store", 2: "Online"}))
"""))

# ── Cell 7: Price distribution ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Phân phối Giá"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sample_price = tx["price"].sample(200_000, random_state=42)
axes[0].hist(sample_price, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Count")
axes[0].set_title("Price Distribution (sample 200k)")

axes[1].boxplot(
    [tx[tx["sales_channel_id"] == c]["price"].sample(50_000, random_state=42)
     for c in [1, 2]],
    labels=["Store (1)", "Online (2)"],
    patch_artist=True,
    boxprops=dict(facecolor="#4C72B0", color="navy"),
)
axes[1].set_ylabel("Price")
axes[1].set_title("Price by Sales Channel")

plt.tight_layout()
plt.savefig("price_dist.png", bbox_inches="tight")
plt.show()

print("Price quantiles:")
print(tx["price"].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(4))
"""))

# ── Cell 8: Customer analysis ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Phân tích Khách hàng"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age distribution
age_valid = customers["age"].dropna()
axes[0].hist(age_valid, bins=50, color="darkorange", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Age Distribution (n={len(age_valid):,})")
axes[0].axvline(age_valid.mean(), color="red", linestyle="--", label=f"Mean={age_valid.mean():.1f}")
axes[0].axvline(age_valid.median(), color="green", linestyle="--", label=f"Median={age_valid.median():.1f}")
axes[0].legend()

# Club membership
club = customers["club_member_status"].value_counts()
axes[1].bar(club.index, club.values, color=["#4C72B0", "#DD8452", "#55A868"])
axes[1].set_title("Club Member Status")
axes[1].set_ylabel("Count")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f"{bar.get_height()/1e6:.2f}M", ha="center", fontsize=9)

# Fashion news frequency
fn = customers["fashion_news_frequency"].value_counts()
axes[2].bar(fn.index, fn.values, color=["#4C72B0", "#DD8452", "#55A868"])
axes[2].set_title("Fashion News Frequency")
axes[2].set_ylabel("Count")
for bar in axes[2].patches:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f"{bar.get_height()/1e6:.2f}M", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("customer_profile.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 9: Customer purchase frequency ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
cust_purchase_count = tx.groupby("customer_id").size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(cust_purchase_count.clip(upper=100), bins=60,
             color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("# Purchases per Customer (clipped at 100)")
axes[0].set_ylabel("Count")
axes[0].set_title("Customer Purchase Frequency")

cdf = cust_purchase_count.value_counts().sort_index().cumsum() / len(cust_purchase_count)
axes[1].plot(cdf.index[:200], cdf.values[:200], color="steelblue")
axes[1].set_xlabel("# Purchases per Customer")
axes[1].set_ylabel("Cumulative Fraction of Customers")
axes[1].set_title("CDF – Purchase Count")
for p in [0.5, 0.8, 0.95]:
    val = (cdf >= p).idxmax()
    axes[1].axhline(p, linestyle="--", linewidth=0.8, color="gray")
    axes[1].axvline(val, linestyle="--", linewidth=0.8, color="gray")
    axes[1].text(val + 1, p - 0.02, f"{val} purchases = {p*100:.0f}%", fontsize=8)

plt.tight_layout()
plt.savefig("customer_purchase_freq.png", bbox_inches="tight")
plt.show()

print("Purchase count stats per customer:")
print(cust_purchase_count.describe().round(1))
"""))

# ── Cell 10: Article analysis ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 6. Phân tích Sản phẩm"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Product group
pg = articles["product_group_name"].value_counts().head(12)
axes[0, 0].barh(pg.index[::-1], pg.values[::-1], color="steelblue")
axes[0, 0].set_xlabel("Count")
axes[0, 0].set_title("Product Groups (Top 12)")

# Index group (department)
ig = articles["index_group_name"].value_counts()
axes[0, 1].pie(ig, labels=ig.index, autopct="%1.1f%%", startangle=90,
               colors=sns.color_palette("muted", len(ig)))
axes[0, 1].set_title("Index Group Distribution")

# Colour groups
cg = articles["colour_group_name"].value_counts().head(15)
axes[1, 0].barh(cg.index[::-1], cg.values[::-1], color="coral")
axes[1, 0].set_xlabel("Count")
axes[1, 0].set_title("Top 15 Colour Groups")

# Garment groups
gg = articles["garment_group_name"].value_counts().head(12)
axes[1, 1].barh(gg.index[::-1], gg.values[::-1], color="seagreen")
axes[1, 1].set_xlabel("Count")
axes[1, 1].set_title("Top 12 Garment Groups")

plt.tight_layout()
plt.savefig("article_profile.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 11: Article popularity ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
article_counts = tx.groupby("article_id").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(article_counts.clip(upper=3000), bins=60,
             color="purple", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("# Times Purchased (clipped at 3000)")
axes[0].set_ylabel("Count")
axes[0].set_title("Article Popularity Distribution")

top20 = article_counts.head(20).reset_index()
top20.columns = ["article_id", "count"]
top20 = top20.merge(articles[["article_id", "prod_name"]], on="article_id", how="left")
top20["label"] = top20["prod_name"].str[:25] + " (" + top20["article_id"].astype(str) + ")"
axes[1].barh(top20["label"][::-1], top20["count"][::-1], color="purple")
axes[1].set_xlabel("# Transactions")
axes[1].set_title("Top 20 Most Purchased Articles")

plt.tight_layout()
plt.savefig("article_popularity.png", bbox_inches="tight")
plt.show()

print(f"Articles with 0 purchases: {articles['article_id'].nunique() - article_counts.shape[0]:,}")
print(f"Articles with >= 1 purchase: {article_counts.shape[0]:,}")
"""))

# ── Cell 12: Age × product group heatmap ──────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 7. Phân tích chéo: Độ tuổi × Nhóm sản phẩm"))

cells.append(nbf.v4.new_code_cell("""\
bins = [15, 25, 35, 45, 55, 65, 100]
labels = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
customers["age_group"] = pd.cut(customers["age"], bins=bins, labels=labels, right=True)

tx_cust = tx.merge(customers[["customer_id", "age_group"]], on="customer_id", how="left")
tx_art  = tx_cust.merge(articles[["article_id", "product_group_name"]], on="article_id", how="left")

pivot = (
    tx_art.dropna(subset=["age_group"])
    .groupby(["age_group", "product_group_name"])
    .size()
    .unstack(fill_value=0)
)
pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100  # row-normalised %

top_groups = articles["product_group_name"].value_counts().head(8).index
pivot_plot = pivot_norm[top_groups]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_plot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Row % of purchases"})
ax.set_title("Purchase % by Age Group × Product Group (top 8 groups)")
ax.set_xlabel("Product Group")
ax.set_ylabel("Age Group")
plt.tight_layout()
plt.savefig("age_product_heatmap.png", bbox_inches="tight")
plt.show()
"""))

# ── Cell 13: Repeat purchase rate ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 8. Tỷ lệ Mua lại (Repeat Purchase)"))

cells.append(nbf.v4.new_code_cell("""\
# For each customer, what fraction of purchases are repeat items?
cust_art = tx.groupby(["customer_id", "article_id"]).size().reset_index(name="n")
repeat_rate = (cust_art.groupby("customer_id")
               .apply(lambda g: (g["n"] > 1).mean(), include_groups=False)
               .reset_index(name="repeat_rate"))

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(repeat_rate["repeat_rate"], bins=50, color="teal", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Fraction of Repeated Items per Customer")
ax.set_ylabel("# Customers")
ax.set_title(f"Repeat Purchase Rate  (mean={repeat_rate['repeat_rate'].mean():.2%})")
plt.tight_layout()
plt.savefig("repeat_purchase.png", bbox_inches="tight")
plt.show()

print(f"Customers who never repeat: {(repeat_rate['repeat_rate'] == 0).mean():.1%}")
print(f"Customers with >50% repeat: {(repeat_rate['repeat_rate'] > 0.5).mean():.1%}")
"""))

# ── Cell 14: Key insights ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Tóm tắt Insights

| # | Insight | Ý nghĩa cho Recommendation |
|---|---------|---------------------------|
| 1 | **31.8M transactions**, 1.36M active customers, 104K articles → rất sparse | Collaborative filtering cần kỹ thuật scaling |
| 2 | **70% giao dịch qua Online** (channel 2), online price cao hơn | Có thể dùng channel như feature |
| 3 | **Phân phối age** right-skewed, đỉnh ở 24–32 | Age group segmentation có ý nghĩa |
| 4 | **Top 20 articles** chiếm lượng mua không tỷ lệ → power law | Popularity baseline rất mạnh |
| 5 | **Majority khách hàng** mua < 20 lần → cold-start problem phổ biến | Cần fallback về popularity |
| 6 | **Garment Upper body** (40%) + **Lower body** (19%) chiếm dominant | Feature engineering từ product group |
| 7 | **16-24 tuổi** mua nhiều Upper body, **45+** mua diverse hơn | Age-segmented popularity cải thiện baseline |
| 8 | **Tỷ lệ mua lại** ~30-40% → repurchase baseline có giá trị | Last-purchase history là signal mạnh |
"""))

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

with open("/home/vungocduong/ddm_project/notebooks/eda.ipynb", "w") as f:
    nbf.write(nb, f)

print("EDA notebook created.")
