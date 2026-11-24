import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):

    df = df.copy()

    # Implement the seven assignment-specific
    # corrections here.

    return df


def prepare_features(df):

    df = clean_data(df)

    X = df.drop(columns=["waste_kg"])
    y = df["waste_kg"]

    return X, y