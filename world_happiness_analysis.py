# -----------------------------
# 1) Imports
# -----------------------------
import kagglehub
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110


# -----------------------------
# 2) Download Dataset
# -----------------------------
# World Happiness Report
# https://www.kaggle.com/datasets/unsdsn/world-happiness
dataset_path = kagglehub.dataset_download(
    "unsdsn/world-happiness"
)

print(f"Dataset downloaded to: {dataset_path}")

all_files = glob.glob(f"{dataset_path}/**/*", recursive=True)
print("\nFiles available:")
for f in all_files:
    if os.path.isfile(f):
        print(" ", f)


# -----------------------------
# 3) Load CSV (pick the most recent year file)
# -----------------------------
csv_files = sorted(glob.glob(f"{dataset_path}/**/*.csv", recursive=True))

# Print all CSVs so we know what we're working with
print("\nCSV files found:")
for f in csv_files:
    print(" ", os.path.basename(f))

# Try to load the most recent year file
file_path = None
for f in csv_files:
    if "2019" in f or "2020" in f or "2021" in f or "2022" in f:
        file_path = f
        break

# Fallback to last file in the list
if file_path is None:
    file_path = csv_files[-1]

print(f"\nLoading: {os.path.basename(file_path)}")
df = pd.read_csv(file_path)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()


# -----------------------------
# 4) Data Cleaning
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())

df_clean = df.copy()

# Standardize column names (different years have different naming)
df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

# Map common column name variations to standard names
# The dataset changes column names across years — this handles it
col_mapping = {}
for col in df_clean.columns:
    if "country" in col or "region" in col.lower():
        if "country" in col:
            col_mapping[col] = "country"
    if "score" in col or "happiness" in col:
        if "score" in col:
            col_mapping[col] = "happiness_score"
    if "gdp" in col or "economy" in col:
        col_mapping[col] = "gdp_per_capita"
    if "health" in col or "life" in col:
        col_mapping[col] = "health_life_expectancy"
    if "freedom" in col:
        col_mapping[col] = "freedom"
    if "generosity" in col:
        col_mapping[col] = "generosity"
    if "trust" in col or "corruption" in col or "perception" in col:
        col_mapping[col] = "corruption_perception"
    if "social" in col or "family" in col:
        col_mapping[col] = "social_support"
    if "rank" in col and "happiness" in col:
        col_mapping[col] = "happiness_rank"

df_clean = df_clean.rename(columns=col_mapping)
print(f"\nStandardized columns: {df_clean.columns.tolist()}")

# Drop rows with missing happiness score
if "happiness_score" in df_clean.columns:
    df_clean = df_clean.dropna(subset=["happiness_score"])

# Drop duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Duplicates removed: {before - len(df_clean)}")
print(f"Clean shape: {df_clean.shape}")

# Quick stats
if "happiness_score" in df_clean.columns:
    print(f"\nHappiness Score range: {df_clean['happiness_score'].min():.2f} — {df_clean['happiness_score'].max():.2f}")
    print(f"Global average: {df_clean['happiness_score'].mean():.2f}")


# -----------------------------
# 5) EDA - Viz 1: Top 20 Happiest Countries
# -----------------------------
if "happiness_score" in df_clean.columns and "country" in df_clean.columns:
    top_20 = df_clean.nlargest(20, "happiness_score")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#f1c40f" if i < 3 else "#3498db" for i in range(len(top_20))]

    bars = ax.barh(
        top_20["country"][::-1],
        top_20["happiness_score"][::-1],
        color=colors[::-1],
        edgecolor="white"
    )

    ax.set_title("Top 20 Happiest Countries in the World", fontweight="bold", fontsize=14)
    ax.set_xlabel("Happiness Score")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\nHappiest country: {top_20.iloc[0]['country']} — Score: {top_20.iloc[0]['happiness_score']:.2f}")


# -----------------------------
# 6) EDA - Viz 2: Bottom 20 (Least Happy Countries)
# -----------------------------
if "happiness_score" in df_clean.columns and "country" in df_clean.columns:
    bottom_20 = df_clean.nsmallest(20, "happiness_score")

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        bottom_20["country"],
        bottom_20["happiness_score"],
        color="#e74c3c",
        edgecolor="white"
    )

    ax.set_title("20 Least Happy Countries", fontweight="bold", fontsize=14)
    ax.set_xlabel("Happiness Score")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    gap = top_20.iloc[0]["happiness_score"] - bottom_20.iloc[0]["happiness_score"]
    print(f"Gap between happiest and least happy: {gap:.2f} points")


# -----------------------------
# 7) EDA - Viz 3: What Drives Happiness? (Factor Comparison)
# -----------------------------
factor_cols = []
for col in ["gdp_per_capita", "social_support", "health_life_expectancy",
            "freedom", "generosity", "corruption_perception"]:
    if col in df_clean.columns:
        factor_cols.append(col)

if len(factor_cols) >= 3:
    factor_means = df_clean[factor_cols].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c"]
    factor_means.plot(kind="barh", ax=ax, color=colors[:len(factor_means)], edgecolor="white")
    ax.set_title("Average Contribution of Each Factor to Happiness", fontweight="bold", fontsize=13)
    ax.set_xlabel("Average Score")

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\nBiggest driver: {factor_means.idxmax()} ({factor_means.max():.3f})")
    print(f"Smallest driver: {factor_means.idxmin()} ({factor_means.min():.3f})")


# -----------------------------
# 8) EDA - Viz 4: GDP vs Happiness Score (scatter)
# -----------------------------
if "gdp_per_capita" in df_clean.columns and "happiness_score" in df_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        df_clean["gdp_per_capita"],
        df_clean["happiness_score"],
        alpha=0.6, s=60, color="#3498db", edgecolors="gray", linewidth=0.5
    )

    # Add trend line
    z = np.polyfit(df_clean["gdp_per_capita"].dropna(), 
                   df_clean.loc[df_clean["gdp_per_capita"].notna(), "happiness_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean["gdp_per_capita"].min(), df_clean["gdp_per_capita"].max(), 100)
    ax.plot(x_line, p(x_line), color="#e74c3c", linestyle="--", linewidth=2, label="Trend line")

    ax.set_title("GDP per Capita vs Happiness Score", fontweight="bold", fontsize=13)
    ax.set_xlabel("GDP per Capita")
    ax.set_ylabel("Happiness Score")
    ax.legend()

    # Label a few interesting countries
    if "country" in df_clean.columns:
        for _, row in df_clean.nlargest(3, "happiness_score").iterrows():
            ax.annotate(row["country"],
                       (row["gdp_per_capita"], row["happiness_score"]),
                       fontsize=8, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

        for _, row in df_clean.nsmallest(3, "happiness_score").iterrows():
            ax.annotate(row["country"],
                       (row["gdp_per_capita"], row["happiness_score"]),
                       fontsize=8, alpha=0.8,
                       xytext=(5, -10), textcoords="offset points")

    plt.tight_layout()
    plt.show()

    corr_val = df_clean["gdp_per_capita"].corr(df_clean["happiness_score"])
    print(f"Correlation (GDP vs Happiness): {corr_val:.3f}")
    print("Strong positive — richer countries tend to be happier, but it's not everything")


# -----------------------------
# 9) EDA - Viz 5: Social Support vs Happiness
# -----------------------------
if "social_support" in df_clean.columns and "happiness_score" in df_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        df_clean["social_support"],
        df_clean["happiness_score"],
        alpha=0.6, s=60, color="#2ecc71", edgecolors="gray", linewidth=0.5
    )

    z = np.polyfit(df_clean["social_support"].dropna(),
                   df_clean.loc[df_clean["social_support"].notna(), "happiness_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean["social_support"].min(), df_clean["social_support"].max(), 100)
    ax.plot(x_line, p(x_line), color="#e74c3c", linestyle="--", linewidth=2, label="Trend line")

    ax.set_title("Social Support vs Happiness Score", fontweight="bold", fontsize=13)
    ax.set_xlabel("Social Support")
    ax.set_ylabel("Happiness Score")
    ax.legend()

    plt.tight_layout()
    plt.show()

    corr_val = df_clean["social_support"].corr(df_clean["happiness_score"])
    print(f"Correlation (Social Support vs Happiness): {corr_val:.3f}")


# -----------------------------
# 10) EDA - Viz 6: Correlation Heatmap (all factors)
# -----------------------------
if len(factor_cols) >= 3 and "happiness_score" in df_clean.columns:
    corr_cols = ["happiness_score"] + factor_cols
    corr = df_clean[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True
    )
    plt.title("Correlation Heatmap — Happiness Factors", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Print sorted correlations with happiness
    happiness_corr = corr["happiness_score"].drop("happiness_score").sort_values(ascending=False)
    print("\nCorrelations with Happiness Score:")
    for feat, val in happiness_corr.items():
        strength = "strong" if abs(val) > 0.5 else "moderate" if abs(val) > 0.3 else "weak"
        print(f"  {feat:30s} {val:+.3f}  ({strength})")


# -----------------------------
# 11) EDA - Viz 7: Where Does UAE Rank? (Middle East Focus)
# -----------------------------
if "country" in df_clean.columns and "happiness_score" in df_clean.columns:
    # Middle East countries (common names in the dataset)
    me_countries = [
        "United Arab Emirates", "Saudi Arabia", "Qatar", "Bahrain", "Kuwait",
        "Oman", "Jordan", "Lebanon", "Iraq", "Egypt", "Israel", "Turkey",
        "Iran", "Syria", "Yemen", "Palestine"
    ]

    me_data = df_clean[df_clean["country"].isin(me_countries)].sort_values(
        "happiness_score", ascending=True
    )

    if len(me_data) >= 3:
        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ["#f1c40f" if c == "United Arab Emirates" else "#3498db"
                  for c in me_data["country"]]

        bars = ax.barh(me_data["country"], me_data["happiness_score"],
                       color=colors, edgecolor="white")

        ax.set_title("Happiness Score — Middle East Countries", fontweight="bold", fontsize=13)
        ax.set_xlabel("Happiness Score")

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}", va="center", fontsize=9)

        # Add global average line
        global_avg = df_clean["happiness_score"].mean()
        ax.axvline(global_avg, color="#e74c3c", linestyle="--", alpha=0.7,
                   label=f"Global avg: {global_avg:.2f}")
        ax.legend()

        plt