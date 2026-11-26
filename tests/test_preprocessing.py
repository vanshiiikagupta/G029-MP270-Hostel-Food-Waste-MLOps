
import pandas as pd

from src.mp270.preprocessing import (
    clean_data,
    prepare_features,
    build_preprocessor
)


def test_preprocessing():

    df = pd.read_csv("data/raw/train.csv")

    # Remove exact duplicates
    df = clean_data(df)

    # Separate target
    y = df["waste_kg"]

    # Prepare features
    X = prepare_features(df)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    print("\n========== PREPROCESSING TEST ==========")

    print("Original records:", len(df))
    print("Feature shape:", X.shape)
    print("Processed shape:", X_processed.shape)
    print("Target shape:", y.shape)

    print("\nPreprocessing successful!")


if __name__ == "__main__":
    test_preprocessing()
