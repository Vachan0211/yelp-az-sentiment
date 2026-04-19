"""
05_star_predictor.py
--------------------
Predicts the star rating (1–5) of a Yelp review from its text alone,
using Logistic Regression and a Random Forest classifier.

WHY this is more useful than the Phase 1 approach:
    The previous model predicted the VADER sentiment label that was
    generated from the same cleaned text — circular, no new information.

    This model predicts the *actual star rating left by the customer*,
    which is an independent ground truth. Businesses can use this to:
        - Flag incoming reviews likely to be 1-2 stars before they post.
        - Identify reviews where the text tone and star rating disagree
          (potential fake/inflated reviews).
        - Rank complaint severity automatically without reading each review.

Models:
    1. Logistic Regression (fast baseline, interpretable)
    2. Random Forest       (captures non-linear patterns)

Features: TF-IDF vectors (top 3,000 terms) from cleaned_text
Target:   stars_y  (1–5 integer star rating)

Usage:
    python src/05_star_predictor.py

Input:  data/vader_preprocessed_output.csv
Output: outputs/star_predictor_report.txt
        outputs/star_predictor_confusion_matrix.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(path: str) -> tuple:
    """
    Load the preprocessed dataset, convert star ratings to integer labels,
    and split into train / test sets.

    Star ratings (stars_y) are the *actual* customer ratings — completely
    independent from VADER labels, making this a genuine supervised task.

    Args:
        path: Path to vader_preprocessed_output.csv

    Returns:
        (X_train, X_test, y_train, y_test, label_names)
    """
    print("Loading data...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["cleaned_text", "stars_y"])
    df["stars_y"] = df["stars_y"].astype(int)

    # Show class balance
    dist = df["stars_y"].value_counts().sort_index()
    print("\nStar rating distribution:")
    for star, count in dist.items():
        bar = "█" * (count // 5_000)
        print(f"  {star}★  {count:>7,}  {bar}")

    print("\nBuilding TF-IDF features...")
    tfidf   = TfidfVectorizer(max_features=3_000, min_df=5, max_df=0.95)
    X       = tfidf.fit_transform(df["cleaned_text"])
    y       = df["stars_y"].values
    labels  = [f"{i}★" for i in sorted(df["stars_y"].unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, labels, tfidf


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    label_names: list[str],
) -> dict:
    """
    Fit a model, predict on the test set, and return performance metrics.

    Args:
        name:        Human-readable model name for display.
        model:       Unfitted scikit-learn estimator.
        X_train/test: TF-IDF feature matrices.
        y_train/test: Star rating label arrays.
        label_names: List of class names for the classification report.

    Returns:
        Dict with 'name', 'accuracy', and 'report' keys.
    """
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=label_names)

    print(f"  Accuracy : {accuracy:.4f}")
    print(report)

    return {"name": name, "model": model, "accuracy": accuracy, "report": report,
            "y_pred": y_pred, "y_test": y_test}


def plot_confusion_matrix(results: list[dict], label_names: list[str]) -> None:
    """
    Plot side-by-side normalised confusion matrices for both models.

    Args:
        results:     List of dicts returned by evaluate_model.
        label_names: Class label strings.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm     = confusion_matrix(res["y_test"], res["y_pred"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, linewidths=0.5, cbar=False,
        )
        ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']:.2%}", fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Star Rating Prediction — Normalised Confusion Matrices", fontsize=15, y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/star_predictor_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to {path}")


def save_report(results: list[dict]) -> None:
    """Write accuracy and classification report for each model to a text file."""
    path = f"{OUTPUT_DIR}/star_predictor_report.txt"
    with open(path, "w") as f:
        for res in results:
            f.write(f"{'='*60}\n")
            f.write(f"Model: {res['name']}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n\n")
            f.write(res["report"])
            f.write("\n")
    print(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_train, X_test, y_train, y_test, labels, _ = prepare_data(
        "data/vader_preprocessed_output.csv"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1_000, C=1.0, random_state=42),
        "Random Forest":        RandomForestClassifier(
                                    n_estimators=100,
                                    max_depth=20,
                                    random_state=42,
                                    n_jobs=-1,
                                ),
    }

    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test, labels)
        results.append(res)

    plot_confusion_matrix(results, labels)
    save_report(results)

    # Key insight summary
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest model: {best['name']} ({best['accuracy']:.2%} accuracy)")
    print(
        "Business value: this model can flag incoming reviews predicted to be "
        "1-2 stars, enabling staff to respond proactively before ratings compound."
    )


if __name__ == "__main__":
    main()
