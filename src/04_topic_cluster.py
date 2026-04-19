"""
04_topic_cluster.py
-------------------
Discovers themes in negative Yelp reviews using two complementary
text mining approaches:

    1. LDA (Latent Dirichlet Allocation)
       - Discovers *topics* as probability distributions over words.
       - Best at finding thematic groupings across the entire corpus.
       - Answers: "What are the underlying complaint themes?"

    2. K-Means Clustering
       - Groups *reviews* by TF-IDF vector similarity.
       - Best at finding which reviews belong together.
       - Answers: "Which reviews share the same complaint pattern?"

Why both? LDA and K-Means are complementary, not redundant:
    - LDA works on word co-occurrence (probabilistic).
    - K-Means works on document vectors in TF-IDF space (geometric).
    Together they cross-validate each other's theme labels.

The optimal number of clusters K is chosen using the Elbow Method
(inertia vs k plot) rather than an arbitrary fixed value.

Usage:
    python src/04_topic_cluster.py

Input:  data/vader_preprocessed_output.csv
Output: outputs/lda_topics.txt
        outputs/elbow_method.png
        outputs/kmeans_cluster_counts.png
        outputs/kmeans_pca_scatter.png
        data/vader_preprocessed_output.csv  (updated with 'cluster' column)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="pastel")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_negative_reviews(path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset and return only negative-sentiment rows.

    Args:
        path: Path to the preprocessed CSV.

    Returns:
        DataFrame of negative reviews with non-null cleaned_text.
    """
    df = pd.read_csv(path)
    df_neg = df[df["vader_sentiment"] == "Negative"].copy()
    df_neg["cleaned_text"] = df_neg["cleaned_text"].astype(str)
    print(f"Negative reviews loaded: {len(df_neg):,}")
    return df_neg


# ---------------------------------------------------------------------------
# LDA Topic Modelling
# ---------------------------------------------------------------------------

def run_lda(df: pd.DataFrame, n_topics: int = 5, n_top_words: int = 10) -> list[tuple]:
    """
    Fit an LDA model on the negative reviews and print the top words per topic.

    Args:
        df:          DataFrame with a 'cleaned_text' column.
        n_topics:    Number of latent topics to extract.
        n_top_words: Number of top words to display per topic.

    Returns:
        A list of (topic_label, [word, ...]) tuples.
    """
    print(f"\nRunning LDA with {n_topics} topics...")

    vectoriser   = CountVectorizer(max_df=0.95, min_df=5, stop_words="english")
    doc_term_mtx = vectoriser.fit_transform(df["cleaned_text"])
    vocab        = vectoriser.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="online",
    )
    lda.fit(doc_term_mtx)

    topics = []
    lines  = []
    for idx, component in enumerate(lda.components_):
        top_words = [vocab[i] for i in component.argsort()[: -n_top_words - 1 : -1]]
        label     = f"Topic {idx + 1}"
        print(f"  {label}: {', '.join(top_words)}")
        topics.append((label, top_words))
        lines.append(f"{label}: {', '.join(top_words)}\n")

    # Save topics to text file for easy reference
    out_path = f"{OUTPUT_DIR}/lda_topics.txt"
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"  Topics saved to {out_path}")

    return topics


# ---------------------------------------------------------------------------
# Elbow Method — choosing K
# ---------------------------------------------------------------------------

def find_optimal_k(
    X_tfidf,
    k_range: range = range(2, 11),
) -> int:
    """
    Run K-Means for each k in k_range, collect inertia, plot the elbow curve,
    and return the elbow point (largest drop in inertia).

    The elbow is the value of k where inertia stops decreasing sharply —
    adding more clusters beyond this point gives diminishing returns.

    Args:
        X_tfidf:  Sparse TF-IDF matrix.
        k_range:  Range of k values to test.

    Returns:
        The optimal k as an integer.
    """
    print("\nRunning elbow method to find optimal K...")
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_tfidf)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}")

    # Identify elbow as the point of greatest second-derivative (curvature)
    diffs  = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    diffs2 = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
    # +2 because we lose two elements from differencing and k_range starts at 2
    optimal_k = list(k_range)[diffs2.index(max(diffs2)) + 2]
    print(f"\n  Elbow point (optimal K) = {optimal_k}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, marker="o", color="#42A5F5", linewidth=2)
    ax.axvline(x=optimal_k, color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Elbow at k={optimal_k}")
    ax.set_title("Elbow Method — Choosing Optimal Number of Clusters", fontsize=14, pad=12)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.legend()
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/elbow_method.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Elbow plot saved to {path}")

    return optimal_k


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Fit K-Means with k clusters on the TF-IDF matrix of negative reviews.
    Prints the top 10 terms per cluster and saves visualisations.

    Args:
        df: DataFrame with 'cleaned_text'.
        k:  Number of clusters (chosen via elbow method).

    Returns:
        (df with 'cluster' column, fitted TfidfVectorizer)
    """
    print(f"\nRunning K-Means with k={k}...")

    tfidf    = TfidfVectorizer(max_features=1_000, stop_words="english")
    X_tfidf  = tfidf.fit_transform(df["cleaned_text"])
    vocab    = tfidf.get_feature_names_out()

    km       = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(X_tfidf)
    df       = df.copy()
    df["cluster"] = clusters

    # Top terms per cluster
    import numpy as np
    dense    = pd.DataFrame(X_tfidf.todense(), columns=vocab)
    grouped  = dense.groupby(clusters).mean()
    print("\n  Top terms per cluster:")
    for i, row in grouped.iterrows():
        top_terms = row.nlargest(10).index.tolist()
        print(f"    Cluster {i}: {', '.join(top_terms)}")

    # Chart 1 — reviews per cluster
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="cluster", data=df, palette="pastel", ax=ax, edgecolor="white")
    ax.set_title("Negative Reviews per Cluster", fontsize=14, pad=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/kmeans_cluster_counts.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved {path}")

    # Chart 2 — 2-D PCA scatter
    print("  Generating PCA scatter (may take a moment)...")
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap="Set2", alpha=0.5, s=8,
    )
    ax.set_title("2-D PCA — K-Means Clusters of Negative Reviews", fontsize=14, pad=12)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/kmeans_pca_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    return df, tfidf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path  = "data/vader_preprocessed_output.csv"

    df_neg = load_negative_reviews(input_path)

    # --- LDA ---
    run_lda(df_neg, n_topics=5)

    # --- K-Means with data-driven K selection ---
    tfidf_for_elbow = TfidfVectorizer(max_features=1_000, stop_words="english")
    X_elbow         = tfidf_for_elbow.fit_transform(df_neg["cleaned_text"])
    optimal_k       = find_optimal_k(X_elbow)

    df_clustered, _ = run_kmeans(df_neg, k=optimal_k)

    # Persist cluster assignments back to the full preprocessed file
    full_df = pd.read_csv(input_path)
    full_df = full_df.merge(
        df_clustered[["review_id", "cluster"]],
        on="review_id",
        how="left",
    )
    full_df.to_csv(input_path, index=False)
    print(f"\nCluster labels saved back to {input_path}")
    print("Topic modelling and clustering complete.")


if __name__ == "__main__":
    main()
