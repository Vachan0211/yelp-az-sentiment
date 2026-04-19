"""
03_sentiment_eda.py
-------------------
Exploratory data analysis and sentiment visualisation for the
Arizona Yelp dataset.

Charts produced (saved to outputs/):
    1. star_rating_distribution.png
    2. top10_most_reviewed_businesses.png
    3. top10_business_categories.png
    4. sentiment_distribution_pie.png
    5. sentiment_top5_businesses.png
    6. negative_sentiment_ratio_by_category.png
    7. top_tfidf_keywords_negative.png

Usage:
    python src/03_sentiment_eda.py

Input:  data/vader_preprocessed_output.csv
Output: outputs/*.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent visual style
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"Positive": "#4CAF50", "Negative": "#E53935", "Neutral": "#FFC107"}


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} reviews, {df['business_id'].nunique():,} businesses")
    return df


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------

def plot_star_distribution(df: pd.DataFrame) -> None:
    """Bar chart of star rating counts (stars_y = customer star rating)."""
    counts = df["stars_y"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("Blues_d", len(counts)), edgecolor="white")
    ax.set_title("Distribution of Star Ratings", fontsize=14, pad=12)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(counts.index, rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/star_rating_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_top_businesses(df: pd.DataFrame, n: int = 10) -> None:
    """Horizontal bar chart of the n most-reviewed businesses."""
    top = df.groupby("name")["review_id"].count().nlargest(n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    top.plot(kind="barh", ax=ax, color="#42A5F5", edgecolor="white")
    ax.set_title(f"Top {n} Most Reviewed Businesses", fontsize=14, pad=12)
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/top10_most_reviewed_businesses.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_top_categories(df: pd.DataFrame, n: int = 10) -> None:
    """Horizontal bar chart of most common business categories."""
    # Categories are pipe-separated in the 'categories' column
    cat_series = (
        df["categories"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    top_cats = cat_series.value_counts().nlargest(n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    top_cats.plot(kind="barh", ax=ax, color="#FFA726", edgecolor="white")
    ax.set_title(f"Top {n} Business Categories", fontsize=14, pad=12)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/top10_business_categories.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_sentiment_pie(df: pd.DataFrame) -> None:
    """Pie chart of overall VADER sentiment distribution."""
    counts = df["vader_sentiment"].value_counts()
    colors = [PALETTE.get(l, "#9E9E9E") for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Sentiment Distribution (VADER)", fontsize=14, pad=16)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sentiment_distribution_pie.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_top5_sentiment_breakdown(df: pd.DataFrame) -> None:
    """Stacked bar: positive vs negative review count for the 5 most-reviewed businesses."""
    top5 = df.groupby("name")["review_id"].count().nlargest(5).index
    subset = df[df["name"].isin(top5) & (df["vader_sentiment"] != "Neutral")]

    pivot = (
        subset.groupby(["name", "vader_sentiment"])
        .size()
        .unstack(fill_value=0)
        [["Positive", "Negative"]]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[PALETTE["Positive"], PALETTE["Negative"]],
               edgecolor="white")
    ax.set_title("Sentiment Breakdown — Top 5 Most Reviewed Businesses", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Sentiment")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sentiment_top5_businesses.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_negative_ratio_by_category(df: pd.DataFrame, n: int = 10) -> None:
    """Horizontal bar chart of the categories with the highest negative-sentiment ratio."""
    df_exploded = df.copy()
    df_exploded["category"] = df_exploded["categories"].str.split(",")
    df_exploded = df_exploded.explode("category")
    df_exploded["category"] = df_exploded["category"].str.strip()

    cat_total = df_exploded.groupby("category").size()
    cat_neg   = df_exploded[df_exploded["vader_sentiment"] == "Negative"].groupby("category").size()
    neg_ratio = (cat_neg / cat_total).dropna().sort_values(ascending=False)

    # Only show categories with enough reviews to be meaningful
    min_reviews = 100
    valid_cats  = cat_total[cat_total >= min_reviews].index
    neg_ratio   = neg_ratio[neg_ratio.index.isin(valid_cats)].nlargest(n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    neg_ratio.plot(kind="barh", ax=ax, color="#EF5350", edgecolor="white")
    ax.set_title(f"Top {n} Categories by Negative Sentiment Ratio", fontsize=14, pad=12)
    ax.set_xlabel("Negative Sentiment Ratio")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/negative_sentiment_ratio_by_category.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_tfidf_keywords_negative(df: pd.DataFrame, n: int = 15) -> None:
    """Bar chart of the top TF-IDF terms in negative reviews."""
    neg_texts = df[df["vader_sentiment"] == "Negative"]["cleaned_text"].dropna()

    tfidf    = TfidfVectorizer(max_features=500, stop_words="english")
    matrix   = tfidf.fit_transform(neg_texts)
    scores   = matrix.mean(axis=0).A1
    terms    = tfidf.get_feature_names_out()

    top_df = (
        pd.Series(scores, index=terms)
        .nlargest(n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    top_df.plot(kind="barh", ax=ax, color="#42A5F5", edgecolor="white")
    ax.set_title(f"Top {n} TF-IDF Keywords in Negative Reviews", fontsize=14, pad=12)
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_ylabel("")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/top_tfidf_keywords_negative.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data("data/vader_preprocessed_output.csv")

    print("\nGenerating charts...")
    plot_star_distribution(df)
    plot_top_businesses(df)
    plot_top_categories(df)
    plot_sentiment_pie(df)
    plot_top5_sentiment_breakdown(df)
    plot_negative_ratio_by_category(df)
    plot_tfidf_keywords_negative(df)

    print(f"\nAll charts saved to /{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
