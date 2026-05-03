"""Append missing EDA sections to the existing eda.ipynb."""

import json
from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).parent / "eda.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

extra_cells = []

# ── Section 10: Seasonal Analysis ─────────────────────────────────────────────
extra_cells.append(nbf.v4.new_markdown_cell("## 10. Phân tích Mùa vụ (Seasonality)"))

extra_cells.append(nbf.v4.new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
                                        
ROOT = Path.cwd().resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)                                                                    

sns.set_theme(style="white", palette="muted", font_scale=1.1)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
                                  
plt.rcParams["figure.dpi"] = 110

DATA = Path("../data")


articles     = pd.read_parquet(DATA / "articles.parquet")
transactions = pd.read_parquet(DATA / "transactions.parquet")
"""))

extra_cells.append(nbf.v4.new_code_cell("""\
# Mean price by product group per month
tx_temp = transactions.copy()
tx_temp["month_start"] = tx_temp["t_dat"].dt.to_period("M").dt.to_timestamp()
df_mg = tx_temp.merge(articles[["article_id", "product_group_name"]], on="article_id")

target_groups = [
    "Garment Upper body", "Garment Lower body",
    "Garment Full body", "Accessories", "Underwear", "Shoes",
]
filtered = df_mg[df_mg["product_group_name"].isin(target_groups)]
grouped = (
    filtered.groupby(["product_group_name", "month_start"])["price"]
    .agg(["mean", "count", "std"])
    .reset_index()
)
grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
grouped["ci_lower"] = grouped["mean"] - 1.96 * grouped["se"]
grouped["ci_upper"] = grouped["mean"] + 1.96 * grouped["se"]

colors = ["#1f77b4", "#ffa500", "#00ced1", "#d62728", "#20b2aa", "#9467bd"]
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()
for i, grp in enumerate(target_groups):
    ax = axes[i]
    gd = grouped[grouped["product_group_name"] == grp]
    ax.plot(gd["month_start"], gd["mean"], color=colors[i], linewidth=2.5)
    ax.fill_between(gd["month_start"], gd["ci_lower"], gd["ci_upper"],
                    color=colors[i], alpha=0.2)
    ax.set_title(f"Mean price – {grp}")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
plt.suptitle("Mean Price per Month by Product Group (95% CI)", y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "seasonal_price.png", bbox_inches="tight")
plt.show()
"""))

extra_cells.append(nbf.v4.new_code_cell("""\
# PCA + GMM clustering on monthly sales patterns
tx_temp["month_year"] = tx_temp["t_dat"].dt.to_period("M")
df_cat = tx_temp.merge(
    articles[["article_id", "index_group_name", "index_name", "product_type_name"]],
    on="article_id"
)

df_cat["Category"] = (
    df_cat["index_group_name"] + " | " +
    df_cat["index_name"] + " | " +
    df_cat["product_type_name"]
)

cat_totals = df_cat.groupby("Category").size()
monthly_cat = df_cat.groupby(["Category", "month_year"]).size().reset_index(name="cnt")

monthly_cat["pct"] = monthly_cat.apply(
    lambda r: 100 * r["cnt"] / cat_totals[r["Category"]], axis=1
)
monthly_cat["my_str"] = monthly_cat["month_year"].astype(str)
pca_data = monthly_cat.pivot(index="Category", columns="my_str", values="pct").fillna(0)

scaled = StandardScaler().fit_transform(pca_data)
pcs = PCA(n_components=2, random_state=42).fit_transform(scaled)

gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(pcs)
pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Season": labels})
season_map = {0: "No season", 1: "Spring/Summer", 2: "Autumn/Winter", 3: "Mixed"}

fig, ax = plt.subplots(figsize=(10, 6)) 

for s, name in season_map.items():
    sub = pca_df[pca_df["Season"] == s]
    ax.scatter(sub["PC1"], sub["PC2"], label=name, s=25, alpha=0.7, edgecolors='w')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of Category Sales Trends – GMM Clusters (Seasonality)", pad=15)

ax.legend(
    title="Seasonality Type", 
    loc='upper left', 
    bbox_to_anchor=(1.02, 1), 
    borderaxespad=0,
    frameon=True
)

sns.despine()

plt.tight_layout()
plt.savefig(FIG_DIR / "pca_seasonality.png", bbox_inches="tight", dpi=300)
plt.show()

season_counts = pca_df["Season"].value_counts().rename(index=season_map)
print("-" * 30)
print("Thống kê số lượng Category theo mùa:")
print(season_counts)
print("-" * 30)
"""))

# ── Section 11: Out-of-stock ───────────────────────────────────────────────────
extra_cells.append(nbf.v4.new_markdown_cell("## 11. Sản phẩm Out-of-Stock (Discontinued)"))

extra_cells.append(nbf.v4.new_code_cell("""\
tx_temp2 = transactions.copy()
tx_temp2["before_2019"] = tx_temp2["t_dat"] < "2019-01-01"

total_sales = tx_temp2.groupby("article_id").size()
old_sales   = tx_temp2[tx_temp2["before_2019"]].groupby("article_id").size()
ratio = (old_sales / total_sales).fillna(0).rename("before_2019_ratio")

article_sales = pd.DataFrame({"total_sales": total_sales, "before_2019_ratio": ratio})
obsolete = article_sales[article_sales["before_2019_ratio"] >= 0.95]

print(f"Tổng articles trong transactions : {len(article_sales):,}")
print(f"Obsolete (≥95% doanh số trước 2019): {len(obsolete):,}  ({len(obsolete)/len(article_sales):.1%})")

# 1. gridspec_kw={'width_ratios': [1, 1.5]} 
fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.2]})

# --- SUBPLOT 1: HISTOGRAM ---
axes[0].hist(article_sales["before_2019_ratio"], bins=50, color="steelblue",
             edgecolor="white", linewidth=0.3)
axes[0].axvline(0.95, color="red", linestyle="--", label="Threshold 0.95")
axes[0].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

axes[0].set_xlabel("Fraction of sales before 2019")
axes[0].set_ylabel("Number of Articles")
axes[0].set_title("Distribution of Old-Sales Ratio")

# legend
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)

# --- SUBPLOT 2: PIE CHART ---
sizes = [len(obsolete), len(article_sales) - len(obsolete)]
axes[1].pie(sizes, labels=["Discontinued", "Active"],
            autopct="%1.1f%%", colors=["salmon", "steelblue"], 
            startangle=90, pctdistance=0.85, explode=(0.05, 0))
axes[1].set_title("Active vs Discontinued Articles")

plt.tight_layout()
plt.savefig(FIG_DIR / "out_of_stock.png", bbox_inches="tight")
plt.show()
"""))

# ── Section 12: Repurchase Granularity ────────────────────────────────────────
extra_cells.append(nbf.v4.new_markdown_cell("## 12. Repurchase Rate theo Granularity"))

extra_cells.append(nbf.v4.new_code_cell("""\
df_rep = transactions.merge(
    articles[["article_id", "product_code", "product_group_name"]],
    on="article_id", how="left"
)

def repurchase_pct(df: pd.DataFrame, item_col: str, weeks: list[int] = [1, 2, 3]) -> list[float]:
    df = df.sort_values(["customer_id", item_col, "t_dat"]).copy()
    df["next_purchase"] = df.groupby(["customer_id", item_col])["t_dat"].shift(-1)
    df["days_to_next"]  = (df["next_purchase"] - df["t_dat"]).dt.days
    total = len(df)
    return [
        100 * ((df["days_to_next"] > 0) & (df["days_to_next"] <= 7 * w)).sum() / total
        for w in weeks
    ]

weeks = [1, 2, 3]
article_rep  = repurchase_pct(df_rep, "article_id",        weeks)
product_rep  = repurchase_pct(df_rep, "product_code",      weeks)
category_rep = repurchase_pct(df_rep, "product_group_name", weeks)

plot_df = pd.DataFrame(
    {"Article ID": article_rep, "Product Code": product_rep, "Category": category_rep},
    index=[f"Within week {w}" for w in weeks],
)

x = np.arange(len(plot_df.index))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
for i, col in enumerate(plot_df.columns):
    ax.bar(x + (i - 1) * width, plot_df[col], width, label=col)
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index)
ax.set_ylabel("Repurchase Rate (%)")
ax.set_title("Repurchase Rate by Granularity & Time Window")
ax.legend(title="Granularity")
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "repurchase_granularity.png", bbox_inches="tight")
plt.show()

print(plot_df.round(2))
"""))

# ── Section 13: Word Cloud ─────────────────────────────────────────────────────
extra_cells.append(nbf.v4.new_markdown_cell("## 13. Word Cloud – Mô tả Sản phẩm"))

extra_cells.append(nbf.v4.new_code_cell("""\
try:
    from wordcloud import WordCloud, STOPWORDS
    import re

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\\s]", " ", text)
        text = re.sub(r"\\s+", " ", text)
        return text

    corpus = " ".join(
        articles["detail_desc"].dropna().astype(str).apply(clean_text)
    )
    stop_words = set(STOPWORDS) | {"cm", "product", "item", "size", "wear", "made"}

    wc = WordCloud(
        width=1200, height=500, background_color="white",
        stopwords=stop_words, max_words=200, collocations=False,
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud – Article Detail Descriptions", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wordcloud.png", bbox_inches="tight")
    plt.show()
except ImportError:
    print("wordcloud not installed — run: uv add wordcloud")
"""))

nb.cells.extend(extra_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Added {len(extra_cells)} cells to {NB_PATH}")
