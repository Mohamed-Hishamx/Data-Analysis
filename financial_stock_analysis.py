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
# S&P 500 Stock Data
# https://www.kaggle.com/datasets/camnugent/sandp500
dataset_path = kagglehub.dataset_download(
    "camnugent/sandp500"
)

print(f"Dataset downloaded to: {dataset_path}")

all_files = glob.glob(f"{dataset_path}/**/*", recursive=True)
print("\nFiles available:")
for f in all_files:
    if os.path.isfile(f):
        print(" ", f)


# -----------------------------
# 3) Load CSV
# -----------------------------
csv_files = glob.glob(f"{dataset_path}/**/*.csv", recursive=True)

# Try to find the main combined file
file_path = None
for f in csv_files:
    if "all_stocks" in f.lower() or "sp500" in f.lower() or "sandp" in f.lower():
        file_path = f
        break

if file_path is None:
    # Pick the largest CSV (likely the combined one)
    file_path = max(csv_files, key=os.path.getsize)

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

# Standardize column names
df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

# Parse date
date_col = None
for col in df_clean.columns:
    if "date" in col:
        date_col = col
        break

if date_col:
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[date_col])
    df_clean["year"] = df_clean[date_col].dt.year
    df_clean["month"] = df_clean[date_col].dt.month
    df_clean["day_of_week"] = df_clean[date_col].dt.day_name()

# Find key columns
close_col = None
volume_col = None
name_col = None

for col in df_clean.columns:
    if "close" in col and "adj" not in col:
        close_col = col
    elif "volume" in col:
        volume_col = col
    elif "name" in col or "ticker" in col or "symbol" in col:
        name_col = col

print(f"\nDate: {date_col} | Close: {close_col} | Volume: {volume_col} | Stock: {name_col}")

# Make sure numeric columns are actually numeric
for col in [close_col, volume_col, "open", "high", "low"]:
    if col and col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Drop rows with missing close price
if close_col:
    df_clean = df_clean.dropna(subset=[close_col])

# Calculate daily return
if close_col and name_col:
    df_clean = df_clean.sort_values([name_col, date_col])
    df_clean["daily_return"] = df_clean.groupby(name_col)[close_col].pct_change() * 100

# Calculate daily price range (high - low)
if "high" in df_clean.columns and "low" in df_clean.columns:
    df_clean["daily_range"] = df_clean["high"] - df_clean["low"]

# Drop duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Duplicates removed: {before - len(df_clean)}")
print(f"Clean shape: {df_clean.shape}")

if name_col:
    print(f"Unique stocks: {df_clean[name_col].nunique()}")


# -----------------------------
# 5) EDA - Viz 1: Top 10 Stocks by Average Closing Price
# -----------------------------
if close_col and name_col:
    avg_price = (
        df_clean.groupby(name_col)[close_col]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    avg_price[::-1].plot(kind="barh", ax=ax, color="#2ecc71", edgecolor="white")
    ax.set_title("Top 10 Stocks by Average Closing Price", fontweight="bold", fontsize=13)
    ax.set_xlabel("Average Close Price ($)")

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f"${width:,.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"Most expensive stock (avg): {avg_price.index[0]} — ${avg_price.values[0]:,.2f}")


# -----------------------------
# 6) EDA - Viz 2: Top 10 Most Traded Stocks (by Volume)
# -----------------------------
if volume_col and name_col:
    avg_volume = (
        df_clean.groupby(name_col)[volume_col]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    avg_volume[::-1].plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Top 10 Most Traded Stocks (Avg Daily Volume)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Average Daily Volume")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    plt.tight_layout()
    plt.show()

    print(f"Most traded: {avg_volume.index[0]} — {avg_volume.values[0]:,.0f} avg daily volume")


# -----------------------------
# 7) EDA - Viz 3: Price Trend of Top 5 Stocks Over Time
# -----------------------------
if close_col and name_col and date_col:
    # Pick top 5 by average volume (most interesting to track)
    top5_stocks = (
        df_clean.groupby(name_col)[volume_col]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, stock in enumerate(top5_stocks):
        stock_data = df_clean[df_clean[name_col] == stock].sort_values(date_col)
        ax.plot(stock_data[date_col], stock_data[close_col],
                label=stock, linewidth=1.5, alpha=0.85, color=colors[i])

    ax.set_title("Price Trend — Top 5 Most Traded Stocks", fontweight="bold", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 8) EDA - Viz 4: Daily Return Distribution (all stocks combined)
# -----------------------------
if "daily_return" in df_clean.columns:
    returns = df_clean["daily_return"].dropna()

    # Remove extreme outliers for better visualization
    returns = returns[(returns > -10) & (returns < 10)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, color="#3498db", edgecolor="white", alpha=0.85)

    avg_return = returns.mean()
    ax.axvline(avg_return, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean: {avg_return:.3f}%")
    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

    ax.set_title("Daily Return Distribution (All S&P 500 Stocks)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    print(f"Average daily return: {avg_return:.4f}%")
    print(f"Std deviation: {returns.std():.4f}%")
    print(f"Positive days: {(returns > 0).sum() / len(returns) * 100:.1f}%")
    print(f"Negative days: {(returns < 0).sum() / len(returns) * 100:.1f}%")


# -----------------------------
# 9) EDA - Viz 5: Most Volatile Stocks (highest std of daily returns)
# -----------------------------
if "daily_return" in df_clean.columns and name_col:
    volatility = (
        df_clean.groupby(name_col)["daily_return"]
        .std()
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if v > volatility.median() else "#3498db" for v in volatility.values]
    volatility[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.set_title("Top 15 Most Volatile Stocks (Std Dev of Daily Returns)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Volatility (Std Dev %)")

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\nMost volatile: {volatility.index[0]} — {volatility.values[0]:.2f}% daily std dev")
    print(f"Least volatile in top 15: {volatility.index[-1]} — {volatility.values[-1]:.2f}%")


# -----------------------------
# 10) EDA - Viz 6: Average Return by Day of Week
# -----------------------------
if "daily_return" in df_clean.columns and "day_of_week" in df_clean.columns:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_returns = df_clean.groupby("day_of_week")["daily_return"].mean()
    day_returns = day_returns.reindex(day_order).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in day_returns.values]
    bars = ax.bar(day_returns.index, day_returns.values, color=colors, edgecolor="white")
    ax.set_title("Average Daily Return by Day of Week", fontweight="bold", fontsize=13)
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Return (%)")
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.002 if height > 0 else height - 0.008,
                f"{height:.4f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()

    best_day = day_returns.idxmax()
    worst_day = day_returns.idxmin()
    print(f"Best day: {best_day} ({day_returns[best_day]:.4f}%)")
    print(f"Worst day: {worst_day} ({day_returns[worst_day]:.4f}%)")


# -----------------------------
# 11) EDA - Viz 7: Correlation Between Price, Volume, and Volatility
# -----------------------------
if name_col and close_col and volume_col and "daily_return" in df_clean.columns:
    stock_summary = df_clean.groupby(name_col).agg(
        avg_price=(close_col, "mean"),
        avg_volume=(volume_col, "mean"),
        volatility=("daily_return", "std"),
        avg_return=("daily_return", "mean"),
        total_days=(close_col, "count")
    ).reset_index()

    # Only stocks with enough data
    stock_summary = stock_summary[stock_summary["total_days"] >= 100]

    corr_cols = ["avg_price", "avg_volume", "volatility", "avg_return"]
    corr = stock_summary[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation: Price, Volume, Volatility, Returns", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nCorrelation insights:")
    for col in corr_cols:
        if col != "avg_return":
            val = corr.loc["avg_return", col]
            print(f"  avg_return vs {col:15s}: {val:+.3f}")


# -----------------------------
# 12) EDA - Viz 8: Risk vs Return Scatter (per stock)
# -----------------------------
if "avg_return" in stock_summary.columns and "volatility" in stock_summary.columns:
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        stock_summary["volatility"],
        stock_summary["avg_return"],
        s=stock_summary["avg_volume"] / stock_summary["avg_volume"].max() * 300,
        alpha=0.5,
        c=stock_summary["avg_return"],
        cmap="RdYlGn",
        edgecolors="gray",
        linewidth=0.5
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.axvline(x=stock_summary["volatility"].median(), color="gray",
               linestyle="--", alpha=0.5, label="Median volatility")

    ax.set_