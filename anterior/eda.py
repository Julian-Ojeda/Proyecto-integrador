from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub

DATASET_NAME = "uciml/adult-census-income"
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    csv_files = list(Path(dataset_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_path}")
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path, na_values=["?", "", " "])
    return df


def print_summary(df):
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS - Adult Census Income")
    print("=" * 80)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nDuplicated rows:", df.duplicated().sum())
    print("\nTarget distribution:")
    print(df["income"].value_counts(dropna=False, normalize=True).round(4))
    print("\nNumerical summary:")
    print(df.describe().round(2).to_string())
    print("\nCategorical summary:")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))


def save_distribution_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.countplot(data=df, x="income", color="#66c2a5", ax=axes[0])
    axes[0].set_title("Target distribution (income)")
    axes[0].set_xlabel("Income")
    axes[0].set_ylabel("Count")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d")

    age_values = df["age"].dropna()
    sns.histplot(age_values, bins=30, kde=True, color="steelblue", ax=axes[1])
    axes[1].set_title("Age distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_age_distribution.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    numeric_cols = ["age", "fnlwgt", "capital.gain", "hours.per.week"]
    for ax, col in zip(axes.ravel(), numeric_cols):
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax, color="darkorange")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.countplot(data=df, x="sex", hue="income", palette="pastel", ax=axes[0])
    axes[0].set_title("Income by sex")
    axes[0].set_xlabel("Sex")
    axes[0].set_ylabel("Count")

    sns.countplot(data=df, x="race", hue="income", palette="Set2", ax=axes[1])
    axes[1].set_title("Income by race")
    axes[1].set_xlabel("Race")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "income_by_demographics.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    top_jobs = df["occupation"].value_counts().head(12)
    sns.barplot(x=top_jobs.values, y=top_jobs.index, orient="h", color="#4C72B0", ax=ax)
    ax.set_title("Top occupations")
    ax.set_xlabel("Count")
    ax.set_ylabel("Occupation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_occupations.png", dpi=200)
    plt.close(fig)


def save_correlations(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=200)
    plt.close(fig)


def main():
    df = load_data()
    print_summary(df)
    save_distribution_plots(df)
    save_correlations(df)
    print("\nPlots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

