
import pandas as pd

df = pd.read_csv("data/processed/train_cleaned.csv")

text_columns = df.select_dtypes(
    include="object"
).columns

print("SPELLING VARIANT AUDIT")
print("======================")

for column in text_columns:
    print(f"\nColumn: {column}")
    print(df[column].value_counts(dropna=False).to_string())

