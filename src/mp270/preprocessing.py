
import pandas as pd


def clean_data(df):
    """
    Clean the hostel food waste dataset.

    Currently implemented:
    1. Remove exact duplicate rows.

    Other assignment-specific cleaning rules
    will be added after verification.
    """

    df = df.copy()

    print("\n========== DATA CLEANING ==========")

    print("Original rows:", len(df))

    # 1. Remove exact duplicate rows
    duplicates = df.duplicated().sum()

    print("Exact duplicates found:", duplicates)

    df = df.drop_duplicates().copy()

    print("Rows after removing duplicates:", len(df))

    # 2. Verify record IDs
    if "record_id" in df.columns:
        print(
            "Remaining duplicate IDs:",
            df["record_id"].duplicated().sum()
        )

    print("\nCleaning completed!")

    return df
