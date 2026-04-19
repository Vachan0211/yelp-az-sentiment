"""
06_financial_health.py
----------------------
Computes a Business Health Score (0–100) for each Arizona business
by combining three signals derived from Yelp review data:

    1. Sentiment ratio  — proportion of positive reviews (VADER)
    2. Average star rating — raw customer score (stars_y)
    3. Review volume    — log-normalised review count (more reviews = stronger signal)

An optional fourth signal (star rating trend) is computed if the
dataset contains a 'date' column, capturing whether sentiment is
improving or declining over time.

A composite health score is then mapped to a risk tier:

    ≥ 70  → Healthy   — performing well, monitor regularly
    50–69 → At-Risk   — declining signals, action recommended
    < 50  → Critical  — consistent negative sentiment, urgent review needed

Financial impact estimation:
    Research by Harvard Business School (Luca, 2016) found that a
    one-star increase on Yelp is associated with a 5–9% revenue uplift
    for independent restaurants. We use a conservative 5% estimate to
    translate the star gap between a business and the market average
    into an estimated annual revenue opportunity.

    This is a directional estimate only — exact figures depend on
    business size, category, and market conditions.

Usage:
    python src/06_financial_health.py

Input:  data/vader_preprocessed_output.csv
Output: outputs/business_health_scores.csv
        outputs/health_score_distribution.png
        outputs/at_risk_businesses.png
        outputs/star_gap_revenue_impact.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# Conservative revenue elasticity (5% per star, per HBS Luca 2016)
REVENUE_ELASTICITY = 0.05

# Assumed average annual revenue for a mid-size Arizona restaurant / retail business
# (U.S. restaurant industry average ~$1.1M/year; we use $800K as a conservative baseline)
BASELINE_REVENUE_USD = 800_000


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level data to the business level and compute
    the composite Business Health Score.

    Score components (each normalised 0–1 before weighting):
        - pos_ratio    : proportion of Positive VADER reviews  (weight 0.40)
        - avg_stars    : mean star_y normalised to [0,1]       (weight 0.40)
        - log_vol      : log10(review_count) / log10(max_vol)  (weight 0.20)

    Final score = weighted sum × 100, rounded to 1 decimal place.

    Args:
        df: Preprocessed DataFrame with 'vader_sentiment', 'stars_y',
            'business_id', 'name', and 'categories'.

    Returns:
        DataFrame with one row per business and a 'health_score' column.
    """
    print("Computing per-business aggregates...")

    grp = df.groupby(["business_id", "name", "categories"])

    # Sentiment ratio (positive reviews / total reviews)
    sentiment_counts = df.groupby(["business_id", "vader_sentiment"]).size().unstack(fill_value=0)
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0
    sentiment_counts["total"]     = sentiment_counts.sum(axis=1)
    sentiment_counts["pos_ratio"] = sentiment_counts["Positive"] / sentiment_counts["total"]

    # Average star rating and review volume
    star_agg = grp["stars_y"].agg(avg_stars="mean", review_count="count").reset_index()

    # Merge
    scores = star_agg.merge(
        sentiment_counts[["pos_ratio", "Negative", "total"]].reset_index(),
        on="business_id",
    )
    scores["neg_ratio"] = scores["Negative"] / scores["total"]

    # Normalise components
    scores["norm_pos"]  = scores["pos_ratio"]                                   # already [0,1]
    scores["norm_star"] = (scores["avg_stars"] - 1) / 4                         # [1,5] → [0,1]
    max_log             = np.log10(scores["review_count"].max())
    scores["norm_vol"]  = np.log10(scores["review_count"].clip(lower=1)) / max_log

    # Weighted composite
    scores["health_score"] = (
        0.40 * scores["norm_pos"]  +
        0.40 * scores["norm_star"] +
        0.20 * scores["norm_vol"]
    ) * 100
    scores["health_score"] = scores["health_score"].round(1)

    # Risk tier
    scores["risk_tier"] = pd.cut(
        scores["health_score"],
        bins=[0, 50, 70, 100],
        labels=["Critical", "At-Risk", "Healthy"],
        include_lowest=True,
    )

    print(f"  Scored {len(scores):,} businesses")
    print("\nRisk tier breakdown:")
    print(scores["risk_tier"].value_counts().to_string())

    return scores


def add_star_trend(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    If 'date' is present, compute whether each business's average star
    rating is trending upward or downward over its last 12 months vs
    the previous 12 months.

    Adds a 'star_trend' column ('Improving', 'Declining', 'Stable').

    Args:
        df:     Full preprocessed DataFrame.
        scores: Business-level scores DataFrame.

    Returns:
        Updated scores DataFrame with 'star_trend'.
    """
    if "date" not in df.columns:
        scores["star_trend"] = "Unknown"
        return scores

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df          = df.dropna(subset=["date"])
    cutoff      = df["date"].max() - pd.DateOffset(months=12)
    cutoff_prev = cutoff - pd.DateOffset(months=12)

    recent = (
        df[df["date"] >= cutoff]
        .groupby("business_id")["stars_y"].mean()
        .rename("recent_avg")
    )
    prev = (
        df[(df["date"] >= cutoff_prev) & (df["date"] < cutoff)]
        .groupby("business_id")["stars_y"].mean()
        .rename("prev_avg")
    )

    trend_df = pd.concat([recent, prev], axis=1).dropna()
    trend_df["delta"] = trend_df["recent_avg"] - trend_df["prev_avg"]
    trend_df["star_trend"] = trend_df["delta"].apply(
        lambda d: "Improving" if d > 0.1 else ("Declining" if d < -0.1 else "Stable")
    )

    scores = scores.merge(
        trend_df[["star_trend"]].reset_index(),
        on="business_id",
        how="left",
    )
    scores["star_trend"] = scores["star_trend"].fillna("Unknown")
    return scores


# ---------------------------------------------------------------------------
# Revenue impact estimation
# ---------------------------------------------------------------------------

def estimate_revenue_impact(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the potential annual revenue uplift if a business improved
    its average star rating to the market average.

    Formula:
        star_gap         = market_avg_stars - business_avg_stars
        uplift_pct       = star_gap × REVENUE_ELASTICITY
        revenue_impact   = BASELINE_REVENUE_USD × uplift_pct

    Businesses already at or above the market average get a 0 impact (no gap).

    Args:
        scores: Business-level scores DataFrame.

    Returns:
        Updated scores DataFrame with 'star_gap' and 'revenue_opportunity_usd'.
    """
    market_avg = scores["avg_stars"].mean()
    scores["star_gap"] = (market_avg - scores["avg_stars"]).clip(lower=0).round(2)
    scores["revenue_opportunity_usd"] = (
        scores["star_gap"] * REVENUE_ELASTICITY * BASELINE_REVENUE_USD
    ).round(0).astype(int)
    print(f"\nMarket average star rating: {market_avg:.2f}")
    print(
        f"Businesses with improvement opportunity: "
        f"{(scores['star_gap'] > 0).sum():,} "
        f"({(scores['star_gap'] > 0).mean():.1%})"
    )
    return scores


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_health_distribution(scores: pd.DataFrame) -> None:
    """Histogram of Business Health Score with risk tier shading."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Background shading for risk tiers
    ax.axvspan(0,  50, alpha=0.08, color="#E53935", label="Critical (<50)")
    ax.axvspan(50, 70, alpha=0.08, color="#FFA726", label="At-Risk (50-70)")
    ax.axvspan(70, 100,alpha=0.08, color="#4CAF50", label="Healthy (>70)")

    sns.histplot(scores["health_score"], bins=40, ax=ax, color="#42A5F5", edgecolor="white")
    ax.set_title("Business Health Score Distribution", fontsize=14, pad=12)
    ax.set_xlabel("Health Score (0–100)")
    ax.set_ylabel("Number of Businesses")
    ax.legend(title="Risk Tier", loc="upper left")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/health_score_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_at_risk_businesses(scores: pd.DataFrame, n: int = 15) -> None:
    """Horizontal bar chart of the most-reviewed businesses that are at-risk or critical."""
    at_risk = (
        scores[scores["risk_tier"].isin(["At-Risk", "Critical"])]
        .nlargest(n, "review_count")
        .sort_values("health_score")
    )

    colors = ["#E53935" if t == "Critical" else "#FFA726" for t in at_risk["risk_tier"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(at_risk["name"], at_risk["health_score"], color=colors, edgecolor="white")
    ax.set_title(f"Top {n} Most-Reviewed At-Risk & Critical Businesses", fontsize=14, pad=12)
    ax.set_xlabel("Health Score")
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color="#E53935", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(x=70, color="#FFA726", linestyle="--", linewidth=1, alpha=0.6)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#E53935", label="Critical"),
        Patch(facecolor="#FFA726", label="At-Risk"),
    ]
    ax.legend(handles=legend_els, loc="lower right")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/at_risk_businesses.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_revenue_opportunity(scores: pd.DataFrame, n: int = 15) -> None:
    """Bar chart of estimated revenue opportunity for highest-gap businesses."""
    top_gap = (
        scores[scores["revenue_opportunity_usd"] > 0]
        .nlargest(n, "revenue_opportunity_usd")
        .sort_values("revenue_opportunity_usd")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_gap["name"],
        top_gap["revenue_opportunity_usd"],
        color="#7E57C2", edgecolor="white",
    )
    ax.set_title(
        f"Estimated Annual Revenue Opportunity — Top {n} Businesses\n"
        "(5% uplift per star gained vs. market average, $800K baseline)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Estimated Revenue Opportunity (USD)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/star_gap_revenue_impact.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading preprocessed data...")
    df = pd.read_csv("data/vader_preprocessed_output.csv")
    print(f"  {len(df):,} reviews loaded")

    scores = compute_health_scores(df)
    scores = add_star_trend(df, scores)
    scores = estimate_revenue_impact(scores)

    # Save full scores table
    out_path = f"{OUTPUT_DIR}/business_health_scores.csv"
    scores.to_csv(out_path, index=False)
    print(f"\nFull scores table saved to {out_path}")

    # Visualisations
    print("\nGenerating charts...")
    plot_health_distribution(scores)
    plot_at_risk_businesses(scores)
    plot_revenue_opportunity(scores)

    # Top 5 critical businesses summary
    print("\n--- Top 5 Critical Businesses (highest review volume) ---")
    critical = (
        scores[scores["risk_tier"] == "Critical"]
        .nlargest(5, "review_count")
        [["name", "health_score", "avg_stars", "neg_ratio", "revenue_opportunity_usd"]]
    )
    print(critical.to_string(index=False))
    print("\nFinancial health analysis complete.")


if __name__ == "__main__":
    main()
