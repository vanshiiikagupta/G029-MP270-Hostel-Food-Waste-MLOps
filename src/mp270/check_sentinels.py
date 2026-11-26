
import pandas as pd

df = pd.read_csv("data/processed/train_cleaned.csv")

sentinel_values = [-999, -1, 999, 9999]

numeric_columns = df.select_dtypes(
    include="number"
).columns

print("SENTINEL VALUE AUDIT")
print("====================")

found = False

for column in numeric_columns:
    counts = df[column].value_counts()

    for value in sentinel_values:
        count = counts.get(value, 0)

        if count > 0:
            found = True
            print(
                f"{column}: {value} appears {count} times"
            )

if not found:
    print("No specified sentinel values found.")

