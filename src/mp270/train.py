import pandas as pd

TRAIN_PATH = "data/raw/train.csv"

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nTarget distribution:")
if "waste_kg" in df.columns:
    print(df["waste_kg"].describe())