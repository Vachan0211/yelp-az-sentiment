# Yelp Arizona — Sentiment Analysis & Business Health Scoring

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![spaCy](https://img.shields.io/badge/spaCy-3.6%2B-09A3D5?logo=spacy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A text mining and machine learning pipeline that transforms 430,000+ Yelp reviews
> into actionable business intelligence — including complaint theme discovery,
> star rating prediction, and a financial health score for every business in Arizona.

---

## Overview

This project was built as part of **BANA 572 — Machine Learning & Text Mining** at
Oregon State University. The goal is to demonstrate how unstructured customer feedback
can be converted into structured, decision-ready insights using NLP and ML techniques.

**Dataset:** [Yelp Open Dataset](https://www.yelp.com/dataset) — 431,709 reviews
from 150,000+ businesses, filtered to Arizona.

### Key Questions Answered

| Question | Technique |
|----------|-----------|
| What is the overall sentiment of Arizona Yelp reviews? | VADER Sentiment Analysis |
| What themes appear in negative reviews? | LDA Topic Modeling |
| Which complaint patterns group similar reviews? | K-Means Clustering (elbow-optimised) |
| Can we predict a review's star rating from text alone? | Logistic Regression + Random Forest |
| Which businesses are financially at-risk based on sentiment? | Business Health Scoring |

---

## Results Snapshot

| Metric | Value |
|--------|-------|
| Reviews analysed | 431,709 |
| Sentiment — Positive | 84.1% |
| Sentiment — Negative | 12.7% |
| Star Predictor Accuracy (LR) | 94% |
| LDA Topics Discovered | 5 |
| At-Risk Businesses Flagged | ~16% of dataset |

---

## Project Structure

```
yelp-az-sentiment/
├── src/
│   ├── 01_merge_data.py          # Merge business + review CSVs, filter AZ
│   ├── 02_preprocess.py          # Clean text, lemmatise, VADER sentiment
│   ├── 03_sentiment_eda.py       # EDA charts and sentiment visualisations
│   ├── 04_topic_cluster.py       # LDA topics + K-Means clustering (elbow method)
│   ├── 05_star_predictor.py      # Predict star rating (1–5) from review text
│   └── 06_financial_health.py    # Business Health Score + revenue impact
├── data/
│   └── README.md                 # Download instructions for raw dataset
├── outputs/                      # All charts and reports (auto-generated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline

Run each script in order. Each script reads from `data/` and writes to `outputs/`.

```
01_merge_data.py
      │
      ▼
02_preprocess.py  ──────────────────────────────────────────────────────────────┐
      │                                                                          │
      ▼                                                                          │
03_sentiment_eda.py                                                              │
      │                                                                          │
      ├──▶  04_topic_cluster.py   (LDA + K-Means with elbow)                    │
      │                                                                          │
      ├──▶  05_star_predictor.py  (LR + Random Forest on stars_y)  ◀────────────┘
      │
      └──▶  06_financial_health.py  (Health Score + Revenue Opportunity)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/yelp-az-sentiment.git
cd yelp-az-sentiment
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add the dataset

Download the Yelp CSVs and place them in `data/` — see [`data/README.md`](data/README.md).

### 4. Run the pipeline

```bash
python src/01_merge_data.py
python src/02_preprocess.py
python src/03_sentiment_eda.py
python src/04_topic_cluster.py
python src/05_star_predictor.py
python src/06_financial_health.py
```

All outputs (charts, CSVs, reports) are saved to `outputs/`.

---

## Methods

### Text Preprocessing (`02_preprocess.py`)

Raw review text is cleaned through a five-step pipeline:

1. **Lowercase + noise removal** — strips URLs, numbers, and punctuation.
2. **Tokenisation** — splits text into individual tokens (NLTK `word_tokenize`).
3. **Stopword removal** — removes high-frequency, low-information words.
4. **Lemmatisation** — reduces each token to its dictionary form using spaCy.
   *(Note: only lemmatisation is applied. Stemming and lemmatisation both normalise
   tokens to a base form but through different mechanisms — applying both would
   produce conflicting, degraded tokens.)*
5. **TF-IDF vectorisation** — top 3,000 terms weighted by importance across the corpus.

### Sentiment Analysis — VADER (`02_preprocess.py`)

[VADER](https://github.com/cjhutto/vaderSentiment) is a rule-based sentiment tool
calibrated for informal, short-form text. Each review receives a compound score:

| Score | Label |
|-------|-------|
| ≥ 0.05 | Positive |
| ≤ −0.05 | Negative |
| between | Neutral |

### Complaint Theme Discovery (`04_topic_cluster.py`)

Two complementary techniques are used to understand negative reviews:

- **LDA Topic Modelling** — discovers themes as probability distributions over words.
  Answers: *"What are the underlying complaint themes in the corpus?"*
- **K-Means Clustering** — groups reviews geometrically in TF-IDF space.
  Answers: *"Which reviews share the same complaint pattern?"*

The optimal number of clusters K is selected via the **elbow method** (plotting
inertia vs. k=2..10 and identifying the point of maximum curvature), rather than
using an arbitrary fixed value.

### Star Rating Prediction (`05_star_predictor.py`)

Predicts the actual customer star rating (1–5) from cleaned review text using
TF-IDF features. This is distinct from sentiment classification:

- The *target* (`stars_y`) is independent ground truth — not derived from VADER.
- Enables downstream tasks like early-warning flagging of likely 1-2 star reviews.
- Helps identify reviews where tone and star rating disagree (fake review detection).

### Business Health Scoring (`06_financial_health.py`)

A composite score (0–100) per business, weighted across three signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Positive sentiment ratio | 40% | Proportion of VADER-Positive reviews |
| Average star rating | 40% | Mean `stars_y`, normalised to [0, 1] |
| Review volume | 20% | Log-normalised review count |

**Revenue opportunity** is estimated using a 5% revenue uplift per star improvement
(per Harvard Business School, Luca 2016) applied to the gap between a business's
average rating and the market average, using a conservative $800K annual revenue baseline.

---

## Business Implications

- **Operations teams** can monitor which locations are trending toward at-risk status
  before negative sentiment compounds.
- **Franchise owners** can benchmark individual locations against the market average
  and identify underperformers with the largest revenue opportunity.
- **Marketing teams** can use the star rating predictor to triage and prioritise
  response to incoming reviews.
- **Category managers** can see which service categories (e.g. Fast Food, Automotive)
  carry the highest negative sentiment ratios and require structural intervention.

---

## Authors

| Name | Role |
|------|------|
| Vachan Thambi Naveen | Data pipeline, topic modelling |
| Hinal Natvar Patel   | Sentiment analysis, visualisations |
| Dong Liu             | Clustering, star predictor |
| Arpana Mallavarupu Felix | Financial health module, report |

BANA 572 — Machine Learning & Text Mining, Oregon State University, 2025

---

## License

This project is for educational purposes. The Yelp dataset is subject to
[Yelp's Dataset Terms of Use](https://www.yelp.com/dataset/documentation/terms).
