"""
02_preprocess.py
----------------
Cleans raw Yelp review text and applies VADER sentiment labelling.

Pipeline:
    1. Lowercase + remove URLs, punctuation, numbers
    2. Tokenise with NLTK
    3. Remove stopwords
    4. Lemmatise with spaCy  ← only one normalisation step (no stemming)
    5. Score each review with VADER → Positive / Negative / Neutral label
    6. Build a TF-IDF matrix (top 3,000 terms) for downstream modelling
    7. Save the enriched DataFrame to vader_preprocessed_output.csv

Usage:
    python src/02_preprocess.py

Input:  data/az_reviews.csv
Output: data/vader_preprocessed_output.csv
"""

import re
import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- one-time downloads (safe to re-run) ---
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("vader_lexicon", quiet=True)

# Load the spaCy English model (run `python -m spacy download en_core_web_sm` once)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # only need tagger

STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs, remove non-alphabetic characters,
    tokenise, remove stopwords, and lemmatise using spaCy.

    NOTE: We use lemmatisation only (no stemming). Stemming and
    lemmatisation both reduce words to a base form, but stemming
    is a crude rule-based chop while lemmatisation uses vocabulary
    context to return a proper dictionary form (e.g. 'better' → 'good').
    Using both would apply two conflicting transforms to the same token.

    Args:
        text: Raw review string.

    Returns:
        A single space-joined string of cleaned, lemmatised tokens.
    """
    # Lowercase and strip noise
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # keep only letters + spaces

    # Tokenise and remove stopwords
    tokens = [t for t in word_tokenize(text) if t not in STOP_WORDS and len(t) > 1]

    # Lemmatise (spaCy reads the token's grammatical context)
    doc = nlp(" ".join(tokens))
    lemmatised = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS]

    return " ".join(lemmatised)


# ---------------------------------------------------------------------------
# VADER sentiment scoring
# ---------------------------------------------------------------------------

_sia = SentimentIntensityAnalyzer()


def get_vader_label(text: str) -> str:
    """
    Return a VADER sentiment label for the given cleaned text.

    Thresholds follow VADER's original paper recommendations:
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        otherwise         → Neutral

    Args:
        text: Cleaned (lemmatised) review string.

    Returns:
        One of 'Positive', 'Negative', or 'Neutral'.
    """
    score = _sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    input_path  = "data/az_reviews.csv"
    output_path = "data/vader_preprocessed_output.csv"

    print("Loading Arizona reviews...")
    df = pd.read_csv(input_path)
    df.dropna(subset=["text"], inplace=True)
    print(f"  {len(df):,} reviews loaded")

    print("Cleaning and lemmatising text (this takes a few minutes)...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    print("Running VADER sentiment analysis...")
    df["vader_sentiment"] = df["cleaned_text"].apply(get_vader_label)

    # Sentiment distribution summary
    dist = df["vader_sentiment"].value_counts(normalize=True).mul(100).round(1)
    print("\nSentiment distribution:")
    for label, pct in dist.items():
        print(f"  {label}: {pct}%")

    # TF-IDF matrix (not saved to disk — too large; recreate when needed)
    print("\nBuilding TF-IDF matrix (top 3,000 features)...")
    tfidf      = TfidfVectorizer(max_features=3_000, min_df=5, max_df=0.95)
    X_tfidf    = tfidf.fit_transform(df["cleaned_text"])
    print(f"  TF-IDF shape: {X_tfidf.shape}")

    print(f"\nSaving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

    # Preview
    print("\nSample (original → cleaned):")
    for _, row in df.sample(3, random_state=1).iterrows():
        print(f"  ORIGINAL : {row['text'][:80]}...")
        print(f"  CLEANED  : {row['cleaned_text'][:80]}...")
        print(f"  SENTIMENT: {row['vader_sentiment']}\n")


if __name__ == "__main__":
    main()
