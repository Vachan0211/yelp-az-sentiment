"""
01_merge_data.py
----------------
Merges the Yelp business and review CSVs, then filters for
Arizona businesses only. Output: az_reviews.csv

Usage:
    python src/01_merge_data.py

Inputs  (place in /data):
    yelp_academic_dataset_business.csv
    yelp_academic_dataset_review.csv

Output:
    data/az_reviews.csv
"""

import pandas as pd


def load_and_merge(business_path: str, review_path: str) -> pd.DataFrame:
    """
    Inner-join the business and review datasets on business_id.

    Args:
        business_path: Path to the business CSV file.
        review_path:   Path to the review CSV file.

    Returns:
        A merged DataFrame containing both business metadata and review text.
    """
    print("Loading business data...")
    businesses = pd.read_csv(business_path)

    print("Loading review data (this may take a moment — 4+ GB file)...")
    reviews = pd.read_csv(review_path)

    print("Merging on business_id...")
    merged = pd.merge(businesses, reviews, on="business_id", how="inner")

    print(f"Merged dataset: {len(merged):,} rows, {merged.shape[1]} columns")
    return merged


def filter_arizona(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the merged DataFrame to Arizona businesses only.

    Args:
        df: The full merged DataFrame.

    Returns:
        A DataFrame containing only rows where state == 'AZ'.
    """
    az = df[df["state"] == "AZ"].copy()
    print(f"Arizona subset: {len(az):,} reviews across {az['business_id'].nunique():,} businesses")
    return az


def main():
    business_path = "data/yelp_academic_dataset_business.csv"
    review_path   = "data/yelp_academic_dataset_review.csv"
    output_path   = "data/az_reviews.csv"

    merged = load_and_merge(business_path, review_path)
    az     = filter_arizona(merged)

    print(f"Saving to {output_path}...")
    az.to_csv(output_path, index=False)
    print("Done.")

    # Quick sanity check
    print("\nColumn overview:")
    print(az.dtypes.to_string())
    print("\nFirst row preview:")
    print(az.head(1).T.to_string())


if __name__ == "__main__":
    main()
