
import os
import pandas as pd
from src.mp270.preprocessing import normalize_categories
# Load original data
df = pd.read_csv("data/raw/train.csv")

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Identify the 11 suspected target typos
ratio = df["waste_kg"] / df["waste_weighed_at_bin"]

typo_mask = ratio.between(9.5, 10.5)

print("Records identified:", typo_mask.sum())

# Correct target values
df.loc[typo_mask, "waste_kg"] = (
    df.loc[typo_mask, "waste_kg"] / 10
)
df = normalize_categories(df)
# Save corrected dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/train_cleaned.csv",
    index=False
)

print("Corrected records:", typo_mask.sum())
print("Final rows:", len(df))
print("Corrected dataset saved successfully!")

print("\nCorrected records:")
print(
    df.loc[typo_mask, [
        "record_id",
        "waste_kg",
        "waste_weighed_at_bin"
    ]].to_string(index=False)
)

