"""Step 3 - EDA on the processed MecDDI table and the MecDDI x DDInter overlap.
Outputs tables (csv) and charts (png) to data/processed/eda_linked/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EDA_OUT, LINK_OUT, OUT

EDA_OUT.mkdir(parents=True, exist_ok=True)
pairs = pd.read_csv(OUT / "mecddi_pairs.csv")
drugs = pd.read_csv(OUT / "mecddi_drugs.csv", dtype=str)
long = pd.read_csv(OUT / "mecddi_pair_mechanisms_long.csv")
COLOR = "#1F3864"


def bar(series, title, fname, horizontal=False, ylabel="count"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if horizontal:
        series.iloc[::-1].plot.barh(ax=ax, color=COLOR)
    else:
        series.plot.bar(ax=ax, color=COLOR)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(EDA_OUT / fname, dpi=150)
    plt.close(fig)


# 1. mechanism categories (a pair can have several, so these overlap)
c = long.category_code.value_counts()
c.to_csv(EDA_OUT / "category_counts.csv")
bar(c, "Pairs per mechanism category (overlapping)", "category_counts.png")
print("Category counts:\n", c)

# 2. labels per pair
k = pairs.n_categories.value_counts().sort_index()
k.to_csv(EDA_OUT / "categories_per_pair.csv")
print("\nCategories per pair:\n", k)

# 3. category co-occurrence (which categories appear together)
hot = pairs[[x for x in pairs.columns if x.startswith("cat_")]]
co = hot.T.dot(hot)
co.to_csv(EDA_OUT / "category_cooccurrence.csv")

# 4. most-connected drugs (hubs)
deg = pd.concat([pairs.drug_a_name, pairs.drug_b_name]).value_counts()
deg.rename_axis("drug").reset_index(name="degree").to_csv(EDA_OUT / "drug_degree.csv", index=False)
print(f"\nMedian interactions per drug: {int(deg.median())}; top 10:\n", deg.head(10))
bar(deg.head(15), "Top 15 drugs by number of interactions", "top_drugs.png", horizontal=True)

# 5. drug metadata
for col, fname in [("therapeutic_class", "therapeutic_class_top15"), ("drug_type", "drug_type")]:
    v = drugs[col].value_counts()
    v.to_csv(EDA_OUT / f"{fname}.csv")
    bar(v.head(15), f"Drugs by {col} (top 15)", f"{fname}.png", horizontal=True)

# 6. overlap with DDInter: severity x mechanism category
lp = LINK_OUT / "linked_pairs.csv"
if lp.exists():
    linked = pd.read_csv(lp)
    s = linked.severity.value_counts()
    s.to_csv(EDA_OUT / "linked_severity_counts.csv")
    bar(s, "Severity of linked pairs (DDInter)", "linked_severity.png")
    ex = linked.assign(category=linked.categories.str.split(";")).explode("category").reset_index(drop=True)
    ct = pd.crosstab(ex.severity, ex.category)
    ct.to_csv(EDA_OUT / "severity_by_category_counts.csv")
    pct = ct.div(ct.sum(axis=1), axis=0).round(3)
    pct.to_csv(EDA_OUT / "severity_by_category_rowpct.csv")
    print("\nSeverity x mechanism category (counts):\n", ct)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pct.columns)), pct.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pct.index)), pct.index)
    ax.set_title("Row-normalized: share of each severity per mechanism category")
    fig.colorbar(im)
    plt.tight_layout()
    fig.savefig(EDA_OUT / "severity_by_category.png", dpi=150)
    plt.close(fig)
else:
    print("\n(No linked_pairs.csv yet - run 02_link_ddinter.py for the severity x mechanism analysis.)")
print(f"\nSaved to {EDA_OUT}")
