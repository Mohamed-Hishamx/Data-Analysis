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
# Netflix Movies and TV Shows
# https://www.kaggle.com/datasets/shivamb/netflix-shows
dataset_path = kagglehub.dataset_download(
    "shivamb/netflix-shows"
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

df = pd.read_csv(file_path)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()


# -----------------------------
# 4) Data Cleaning
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum())
print(f"\nMissing % per column:")
print((df.isnull().sum() / len(df) * 100).round(1))

# director and cast have lots of nulls — fill with "Unknown" instead of dropping
df_clean = df.copy()
df_clean["director"] = df_clean["director"].fillna("Unknown")
df_clean["cast"] = df_clean["cast"].fillna("Unknown")
df_clean["country"] = df_clean["country"].fillna("Unknown")

# Drop rows where date_added is missing (small number)
df_clean = df_clean.dropna(subset=["date_added"])

# Parse date_added
df_clean["date_added"] = pd.to_datetime(df_clean["date_added"].str.strip(), errors="coerce")
df_clean = df_clean.dropna(subset=["date_added"])

# Extract useful date parts
df_clean["year_added"] = df_clean["date_added"].dt.year.astype(int)
df_clean["month_added"] = df_clean["date_added"].dt.month
df_clean["month_name"] = df_clean["date_added"].dt.strftime("%b")

# Clean duration column
# Movies have "XX min", TV Shows have "X Season(s)"
df_clean["duration_num"] = df_clean["duration"].str.extract(r"(\d+)").astype(float)

# Drop duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["title", "type", "release_year"])
print(f"\nDuplicates removed: {before - len(df_clean)}")
print(f"Clean shape: {df_clean.shape}")

# Quick look at type split
print(f"\nContent type split:")
print(df_clean["type"].value_counts())


# -----------------------------
# 5) EDA - Viz 1: Movies vs TV Shows (count + percentage)
# -----------------------------
type_counts = df_clean["type"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
colors = ["#e50914", "#221f1f"]  # Netflix red and black
type_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
axes[0].set_title("Content Count: Movies vs TV Shows", fontweight="bold", fontsize=12)
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

# Pie chart
axes[1].pie(type_counts.values, labels=type_counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, textprops={"fontsize": 12})
axes[1].set_title("Content Split", fontweight="bold", fontsize=12)

plt.tight_layout()
plt.show()

movie_pct = type_counts.get("Movie", 0) / type_counts.sum() * 100
print(f"Movies make up {movie_pct:.1f}% of Netflix content")


# -----------------------------
# 6) EDA - Viz 2: Content Added Per Year (growth over time)
# -----------------------------
yearly = df_clean.groupby(["year_added", "type"]).size().unstack(fill_value=0)

# Filter to reasonable years (2010+)
yearly = yearly[yearly.index >= 2010]

fig, ax = plt.subplots(figsize=(12, 6))
yearly.plot(kind="bar", stacked=True, ax=ax, color=["#e50914", "#221f1f"],
            edgecolor="white", linewidth=0.5)
ax.set_title("Content Added to Netflix Per Year", fontweight="bold", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Titles Added")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(title="Type")

plt.tight_layout()
plt.show()

peak_year = yearly.sum(axis=1).idxmax()
peak_count = yearly.sum(axis=1).max()
print(f"Peak year: {int(peak_year)} — {int(peak_count)} titles added")


# -----------------------------
# 7) EDA - Viz 3: Top 15 Countries Producing Netflix Content
# -----------------------------
# Country column can have multiple countries separated by comma
# We'll split and count each country separately
country_split = df_clean["country"].str.split(",").explode().str.strip()
country_split = country_split[country_split != "Unknown"]

top_countries = country_split.value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 7))
top_countries[::-1].plot(kind="barh", ax=ax, color="#e50914", edgecolor="white")
ax.set_title("Top 15 Countries by Netflix Content", fontweight="bold", fontsize=13)
ax.set_xlabel("Number of Titles")

for p in ax.patches:
    width = p.get_width()
    ax.text(width + 5, p.get_y() + p.get_height() / 2,
            f"{int(width)}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

print(f"\nTop country: {top_countries.index[0]} — {top_countries.values[0]} titles")
print(f"US dominates, followed by India and UK")


# -----------------------------
# 8) EDA - Viz 4: Movie Duration Distribution
# -----------------------------
movies = df_clean[df_clean["type"] == "Movie"].copy()
movies = movies[movies["duration_num"].notna()]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(movies["duration_num"], bins=40, color="#e50914", edgecolor="white", alpha=0.85)

median_dur = movies["duration_num"].median()
ax.axvline(median_dur, color="#221f1f", linestyle="--", linewidth=2,
           label=f"Median: {median_dur:.0f} min")

ax.set_title("Movie Duration Distribution on Netflix", fontweight="bold", fontsize=13)
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Number of Movies")
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

print(f"Median movie length: {median_dur:.0f} minutes")
print(f"Shortest: {movies['duration_num'].min():.0f} min")
print(f"Longest: {movies['duration_num'].max():.0f} min")


# -----------------------------
# 9) EDA - Viz 5: Top Genres (listed_in column)
# -----------------------------
genre_split = df_clean["listed_in"].str.split(",").explode().str.strip()
top_genres = genre_split.value_counts().head(12)

fig, ax = plt.subplots(figsize=(12, 6))
top_genres[::-1].plot(kind="barh", ax=ax, color="#b20710", edgecolor="white")
ax.set_title("Top 12 Genres on Netflix", fontweight="bold", fontsize=13)
ax.set_xlabel("Number of Titles")

plt.tight_layout()
plt.show()

print(f"\nMost common genre: {top_genres.index[0]} — {top_genres.values[0]} titles")


# -----------------------------
# 10) EDA - Viz 6: Content Rating Breakdown
# -----------------------------
rating_counts = df_clean["rating"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
rating_counts.plot(kind="bar", ax=ax, color="#e50914", edgecolor="white")
ax.set_title("Content by Rating", fontweight="bold", fontsize=13)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

top_rating = rating_counts.index[0]
print(f"Most common rating: {top_rating} — {rating_counts.values[0]} titles")


# -----------------------------
# 11) Business Conclusion
# -----------------------------
total_titles = len(df_clean)
total_movies = len(df_clean[df_clean["type"] == "Movie"])
total_shows = len(df_clean[df_clean["type"] == "TV Show"])

print("\n" + "=" * 60)
print("BUSINESS CONCLUSION")
print("=" * 60)
print(f"\nTotal titles analyzed: {total_titles:,}")
print(f"Movies: {total_movies:,} | TV Shows: {total_shows:,}")
print(f"Peak content year: {int(peak_year)}")
print(f"Median movie length: {median_dur:.0f} minutes")

print("\nKey takeaways:")
print(f"1) Netflix is heavily movie-focused ({movie_pct:.0f}% movies vs {100-movie_pct:.0f}% TV shows).")
print("   But TV shows drive more watch time per title — worth tracking engagement, not just count.")
print(f"2) Content additions peaked in {int(peak_year)} then dropped — could be COVID production")
print("   slowdowns or a shift toward fewer but higher-quality originals.")
print("3) US produces the most content by far, but India is second — Netflix is clearly")
print("   investing in regional content for growth markets.")
print("4) International Movies and Dramas are the top genres — global audience prefers")
print("   drama-heavy content over comedy or action.")
print("5) Most content is rated TV-MA or TV-14 — Netflix skews toward adult audiences.")
print("   Family-friendly content is a smaller slice, which could be a gap to fill.")