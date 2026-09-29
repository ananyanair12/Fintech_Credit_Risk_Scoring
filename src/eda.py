"""Quick exploratory data analysis. Saves plots to docs/images/eda_*.png.

Usage:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import DOCS_IMAGES, TARGET
from src.data.split import load_raw


def save(fig, name):
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(DOCS_IMAGES / name, dpi=130)
    plt.close(fig)


def main():
    df = load_raw()
    print(df.describe().T.round(3).to_string())

    rate = df[TARGET].mean()
    print(f"\nDefault rate: {rate:.4f} | imbalance ratio (good:bad) = {(1 - rate) / rate:.1f}:1")

    # 1. target distribution
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No default", "Default"], counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_title(f"Target distribution (default rate {rate:.1%})")
    save(fig, "eda_target.png")

    # 2. missing values
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(miss.index, miss.values)
    ax.set(xlabel="% missing", title="Missing values")
    save(fig, "eda_missing.png")

    # 3. key distributions (log1p for heavy tails)
    cols = ["revolving_utilization", "age", "debt_ratio", "monthly_income"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, c in zip(axes.ravel(), cols):
        vals = df[c].dropna()
        if c != "age":
            vals = np.log1p(vals.clip(lower=0))
        ax.hist(vals, bins=50, color="#4c72b0")
        ax.set_title(c if c == "age" else f"log1p({c})")
    save(fig, "eda_distributions.png")

    # 4. correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, "eda_correlation.png")

    # 5. default rate by delinquency history
    df["ever_late"] = (df[["past_due_30_59", "past_due_60_89", "past_due_90"]] > 0).any(axis=1)
    print("\nDefault rate by any past-due history:")
    print(df.groupby("ever_late")[TARGET].mean().round(4).to_string())
    print(f"\nPlots saved to {DOCS_IMAGES}")


if __name__ == "__main__":
    main()