
import pandas as pd

df = pd.read_csv("data/processed/train_cleaned.csv")

numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:
    print(f"\n--- {column} by hostel ---")

    print(
        df.groupby("hostel")[column].agg(
            ["min", "mean", "max"]
        ).round(3)
    )
