import pandas as pd
from src.mp270.preprocessing import normalize_categories

df = pd.read_csv("data/raw/train.csv")

df = normalize_categories(df)

print("NORMALIZED MEAL VALUES")
print(df["meal"].value_counts(dropna=False))

print("\nHOSTEL VALUES")
print(df["hostel"].value_counts(dropna=False))

print("\nUNIQUE MEAL CATEGORIES:")
print(sorted(df["meal"].dropna().unique()))