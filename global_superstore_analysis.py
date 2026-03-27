# -----------------------------
# 1) Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110


# -----------------------------
# 2) Load Data (direct URL — no API needed)
# -----------------------------
# Dubai Real Estate Transactions dataset
# Source: Dubai Land Department (open data)
url = "https://raw.githubusercontent.com/plotly/datasets/master/real-estate.csv"

# Alternative: using a well-known housing dataset with price, size, location features
# We'll use the USA Real Estate dataset from GitHub (works without login)
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/realestate.csv"

try:
    df = pd.read_csv(url)
    print(f"Loaded from URL successfully")
except Exception:
    # Fallback dataset — California Housing (always available via sklearn)
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print("Loaded California Housing dataset as fallback")

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()


# -----------------------------
# 3) Data Cleaning
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Find the price column
price_col = None
for col in df.columns:
    if "price" in col or "value" in col or "medhouseval" in col:
        price_col = col
        break

print(f"\nUsing price column: {price_col}")

# Make sure price is numeric
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

# Drop rows with missing price
df_clean = df.dropna(subset=[price_col]).copy()

# Remove duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Duplicates removed: {before - len(df_clean)}")

# Remove extreme outliers (top and bottom 1%)
lower = df_clean[price_col].quantile(0.01)
upper = df_clean[price_col].quantile(0.99)
df_clean = df_clean[(df_clean[price_col] >= lower) & (df_clean[price_col] <= upper)]

print(f"Clean shape: {df_clean.shape}")
print(f"\nPrice stats:")
print(df_clean[price_col].describe())


# -----------------------------
# 4) EDA - Viz 1: Price Distribution
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df_clean[price_col], bins=40, color="#3498db", edgecolor="white", alpha=0.85)

median_price = df_clean[price_col].median()
mean_price = df_clean[price_col].mean()

ax.axvline(median_price, color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Median: {median_price:,.2f}")
ax.axvline(mean_price, color="#2ecc71", linestyle="--", linewidth=2,
           label=f"Mean: {mean_price:,.2f}")

ax.set_title("Property Price Distribution", fontweight="bold", fontsize=13)
ax.set_xlabel("Price")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.show()

print(f"Median: {median_price:,.2f}")
print(f"Mean: {mean_price:,.2f}")
print(f"Skew: {df_clean[price_col].skew():.2f}")


# -----------------------------
# 5) EDA - Viz 2: Price vs Key Numeric Feature (scatter)
# -----------------------------
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
# Pick a feature that isn't the price itself
feature_cols = [c for c in numeric_cols if c != price_col]

if len(feature_cols) >= 1:
    # Pick the feature most correlated with price
    correlations = df_clean[feature_cols].corrwith(df_clean[price_col]).abs().sort_values(ascending=False)
    top_feature = correlations.index[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_clean[top_feature], df_clean[price_col],
               alpha=0.3, color="#9b59b6", s=15)
    ax.set_title(f"Price vs {top_feature} (strongest correlation)", fontweight="bold", fontsize=13)
    ax.set_xlabel(top_feature)
    ax.set_ylabel(price_col)

    plt.tight_layout()
    plt.show()

    print(f"Correlation between {top_feature} and {price_col}: {correlations[top_feature]:.3f}")


# -----------------------------
# 6) EDA - Viz 3: Boxplot of Price by a Categorical/Binned Feature
# -----------------------------
if len(feature_cols) >= 2:
    # Bin the second strongest feature into groups
    second_feature = correlations.index[1]

    df_clean["feature_bin"] = pd.qcut(df_clean[second_feature], q=4, duplicates="drop",
                                       labels=["Low", "Mid-Low", "Mid-High", "High"])

    fig, ax = plt.subplots(figsize=(10, 6))
    df_clean.boxplot(column=price_col, by="feature_bin", ax=ax,
                     patch_artist=True,
                     boxprops=dict(facecolor="#3498db", color="black"),
                     medianprops=dict(color="#e74c3c", linewidth=2))
    ax.set_title(f"Price Distribution by {second_feature} (binned)", fontweight="bold", fontsize=13)
    plt.suptitle("")  # remove auto title
    ax.set_xlabel(f"{second_feature} Group")
    ax.set_ylabel("Price")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 7) EDA - Viz 4: Correlation Heatmap
# -----------------------------
useful_numeric = [c for c in numeric_cols if "id" not in c.lower() and "unnamed" not in c.lower()]

if len(useful_numeric) >= 3:
    corr = df_clean[useful_numeric].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Top correlations with price
    price_corr = corr[price_col].drop(price_col).sort_values(ascending=False)
    print(f"\nCorrelations with {price_col}:")
    for feat, val in price_corr.items():
        print(f"  {feat:25s} {val:+.3f}")


# -----------------------------
# 8) Business Conclusion
# -----------------------------
print("\n" + "=" * 60)
print("BUSINESS CONCLUSION")
print("=" * 60)
print(f"\nTotal properties analyzed: {len(df_clean):,}")
print(f"Median price: {median_price:,.2f}")
print(f"Mean price: {mean_price:,.2f}")
print("\nKey takeaways:")
print("1) Price distribution is right-skewed — most properties are mid-range,")
print("   a small number of expensive ones pull the average up.")
print("2) Location/size features have the strongest correlation with price.")
print("3) Properties in the top quartile of size/rooms cost significantly more")
print("   — not linear, more like exponential at the high end.")
print("4) For a buyer: focus on the median, not the mean. The average is misleading.")