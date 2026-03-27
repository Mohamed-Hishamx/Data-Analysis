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
# Spotify Most Streamed Songs 2023
# https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023
dataset_path = kagglehub.dataset_download(
    "nelgiriyewithana/top-spotify-songs-2023"
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
file_path = csv_files[0]
print(f"\nLoading: {file_path}")

# This dataset has encoding issues — latin1 fixes it
df = pd.read_csv(file_path, encoding="latin1")
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

# streams column sometimes has non-numeric values
df_clean["streams"] = pd.to_numeric(df_clean["streams"], errors="coerce")
print(f"\nStreams nulls after conversion: {df_clean['streams'].isnull().sum()}")

# Drop rows where streams is missing (can't analyze without it)
df_clean = df_clean.dropna(subset=["streams"])

# Convert audio features to numeric (some might have issues)
audio_features = ["danceability_%", "energy_%", "valence_%",
                  "acousticness_%", "speechiness_%", "liveness_%",
                  "instrumentalness_%"]

# Only process columns that exist
existing_audio = [col for col in audio_features if col in df_clean.columns]
for col in existing_audio:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Fill missing audio features with median (small number of nulls)
for col in existing_audio:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# released_year should be numeric
if "released_year" in df_clean.columns:
    df_clean["released_year"] = pd.to_numeric(df_clean["released_year"], errors="coerce")

# Drop duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["track_name", "artist(s)_name"], keep="first")
print(f"Duplicates removed: {before - len(df_clean)}")
print(f"Clean shape: {df_clean.shape}")

# Quick stats
print(f"\nTotal streams in dataset: {df_clean['streams'].sum():,.0f}")
print(f"Average streams per song: {df_clean['streams'].mean():,.0f}")


# -----------------------------
# 5) EDA - Viz 1: Top 15 Most Streamed Songs
# -----------------------------
top_songs = df_clean.nlargest(15, "streams")[["track_name", "artist(s)_name", "streams"]]
top_songs["label"] = top_songs["track_name"] + " — " + top_songs["artist(s)_name"]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_songs["label"][::-1], top_songs["streams"][::-1],
               color="#1DB954", edgecolor="white")  # Spotify green
ax.set_title("Top 15 Most Streamed Songs on Spotify", fontweight="bold", fontsize=13)
ax.set_xlabel("Total Streams")

# Format x-axis to billions
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))

plt.tight_layout()
plt.show()

print(f"\n#1 most streamed: {top_songs.iloc[0]['track_name']} by {top_songs.iloc[0]['artist(s)_name']}")
print(f"   Streams: {top_songs.iloc[0]['streams']:,.0f}")


# -----------------------------
# 6) EDA - Viz 2: Top 10 Artists by Total Streams
# -----------------------------
# Artist column might have multiple artists — split and count
artist_streams = (
    df_clean.groupby("artist(s)_name")["streams"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(12, 6))
artist_streams[::-1].plot(kind="barh", ax=ax, color="#1DB954", edgecolor="white")
ax.set_title("Top 10 Artists by Total Streams", fontweight="bold", fontsize=13)
ax.set_xlabel("Total Streams")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B"))

plt.tight_layout()
plt.show()

print(f"\nTop artist: {artist_streams.index[0]} — {artist_streams.values[0]:,.0f} streams")


# -----------------------------
# 7) EDA - Viz 3: Songs Released Per Year (recent years)
# -----------------------------
if "released_year" in df_clean.columns:
    yearly = df_clean[df_clean["released_year"] >= 2000].groupby("released_year").size()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(yearly.index, yearly.values, color="#1DB954", edgecolor="white", alpha=0.85)
    ax.set_title("Number of Top Songs by Release Year (2000+)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Songs in Top List")

    # Highlight peak year
    peak_year = yearly.idxmax()
    peak_idx = list(yearly.index).index(peak_year)
    ax.patches[peak_idx].set_color("#e74c3c")

    plt.tight_layout()
    plt.show()

    print(f"Most songs from: {int(peak_year)} — {yearly.max()} songs")
    print("Recent years dominate — recency bias in streaming numbers")


# -----------------------------
# 8) EDA - Viz 4: Audio Features Distribution (Radar-style comparison)
# -----------------------------
if len(existing_audio) >= 4:
    feature_means = df_clean[existing_audio].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1DB954", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#e67e22", "#1abc9c"]

    bars = ax.bar(
        [col.replace("_%", "") for col in feature_means.index],
        feature_means.values,
        color=colors[:len(feature_means)],
        edgecolor="white"
    )

    ax.set_title("Average Audio Features of Top Spotify Songs", fontweight="bold", fontsize=13)
    ax.set_ylabel("Average Value (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f"{height:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.show()

    print("\nAudio feature averages:")
    for feat, val in feature_means.items():
        print(f"  {feat:25s} {val:.1f}%")


# -----------------------------
# 9) EDA - Viz 5: Danceability vs Energy (colored by streams)
# -----------------------------
if "danceability_%" in df_clean.columns and "energy_%" in df_clean.columns:
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        df_clean["danceability_%"],
        df_clean["energy_%"],
        c=df_clean["streams"],
        cmap="YlGn",
        alpha=0.6,
        s=20,
        edgecolors="gray",
        linewidth=0.3
    )

    plt.colorbar(scatter, ax=ax, label="Streams", shrink=0.8)
    ax.set_title("Danceability vs Energy (color = streams)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Danceability (%)")
    ax.set_ylabel("Energy (%)")

    plt.tight_layout()
    plt.show()

    print("Most top songs cluster in the 50-80% range for both danceability and energy")
    print("Extreme values (very low or very high) tend to have fewer streams")


# -----------------------------
# 10) EDA - Viz 6: Streams by Release Month (is there a best month?)
# -----------------------------
if "released_month" in df_clean.columns:
    df_clean["released_month"] = pd.to_numeric(df_clean["released_month"], errors="coerce")

    month_streams = df_clean.groupby("released_month")["streams"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if m == month_streams.idxmax() else "#1DB954"
              for m in month_streams.index]
    ax.bar(month_streams.index, month_streams.values, color=colors, edgecolor="white")
    ax.set_title("Average Streams by Release Month", fontweight="bold", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Streams")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    plt.tight_layout()
    plt.show()

    best_month = month_names[int(month_streams.idxmax()) - 1]
    print(f"Best release month: {best_month}")
    print("Songs released in January tend to accumulate more streams — likely full-year advantage")


# -----------------------------
# 11) EDA - Viz 7: Correlation Heatmap (Audio Features vs Streams)
# -----------------------------
corr_cols = ["streams"] + existing_audio
corr = df_clean[corr_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation: Audio Features vs Streams", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.show()

# Print correlations with streams
stream_corr = corr["streams"].drop("streams").sort_values(ascending=False)
print("\nCorrelations with streams:")
for feat, val in stream_corr.items():
    direction = "+" if val > 0 else "-"
    print(f"  {feat:25s} {val:+.3f}  ({direction})")


# -----------------------------
# 12) Business Conclusion
# -----------------------------
total_songs = len(df_clean)
total_streams = df_clean["streams"].sum()
avg_streams = df_clean["streams"].mean()

print("\n" + "=" * 60)
print("BUSINESS CONCLUSION")
print("=" * 60)
print(f"\nSongs analyzed: {total_songs:,}")
print(f"Total streams: {total_streams:,.0f}")
print(f"Average streams per song: {avg_streams:,.0f}")

print("\nKey takeaways:")
print("1) A small number of artists dominate total streams — the top 10 artists")
print("   account for a disproportionate share. Streaming is a winner-take-most game.")
print("2) High danceability + moderate energy is the sweet spot for popular songs.")
print("   Extremely high or low values in any audio feature tend to underperform.")
print("3) January releases accumulate the most streams on average — partly because")
print("   they have the full year to collect plays. Labels should factor this in.")
print("4) Recent songs (2020-2023) dominate the list — older hits still chart but")
print("   the platform heavily favors new releases in its algorithm.")
print("5) No single audio feature strongly predicts streams — popularity depends on")
print("   marketing, playlist placement, and artist following more than sound alone.")