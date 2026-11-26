
import pandas as pd

df = pd.read_csv("data/processed/train_cleaned.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print("STUCK SENSOR AUDIT")
print("==================")

found = False

for column in numeric_columns:

    # Identify consecutive records with identical readings
    groups = df[column].ne(df[column].shift()).cumsum()

    runs = df.groupby(groups).agg(
        value=(column, "first"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        records=(column, "size"),
        distinct_days=("date", "nunique")
    )

    # Show runs spanning at least 7 distinct dates
    suspicious = runs[runs["distinct_days"] >= 7]

    if not suspicious.empty:
        found = True
        print(f"\nColumn: {column}")
        print(suspicious.to_string(index=False))

if not found:
    print("No repeated readings across 7 or more distinct dates.")
