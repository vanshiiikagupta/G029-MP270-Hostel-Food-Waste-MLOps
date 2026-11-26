
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_rows", 100)
df = pd.read_csv("data/raw/train.csv")

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicates:", df.duplicated().sum())

print("\n========== UNIQUE VALUES ==========")
for col in df.columns:
    print(f"\n{col}:")
    print("Unique values:", df[col].nunique())
    print("Examples:", df[col].unique()[:10])

print("\n========== TARGET OUTLIERS ==========")
print(
    df.nlargest(10, "waste_kg")[
        [c for c in ["record_id", "waste_kg"] if c in df.columns]
    ]
)

print("\n========== NUMERIC SUMMARY ==========")
print(df.select_dtypes(include="number").describe())
print("\n========== ALL COLUMN NAMES ==========")
for col in df.columns:
    print(col)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== DUPLICATE RECORD IDs ==========")
if "record_id" in df.columns:
    print(df["record_id"].duplicated().sum())

print("\n========== HIGHEST WASTE RECORDS ==========")
print(
    df.nlargest(15, "waste_kg").to_string(index=False)
)
print("\n========== DATA QUALITY SUMMARY ==========")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum().to_string())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nDUPLICATE RECORD IDs:")
if "record_id" in df.columns:
    print(df["record_id"].duplicated().sum())

print("\nREPEATED RECORD DETAILS:")
if "record_id" in df.columns:
    duplicates = df[
        df["record_id"].duplicated(keep=False)
    ]
    print(duplicates.to_string(index=False))

print("\nHIGHEST 15 WASTE VALUES:")
print(
    df.nlargest(15, "waste_kg")[
        ["record_id", "waste_kg"]
    ].to_string(index=False)
)


print("\n========== DUPLICATE ANALYSIS ==========")

# Exact duplicate rows
exact_duplicates = df.duplicated().sum()

print("Exact duplicate rows:", exact_duplicates)

# Duplicate record IDs
duplicate_ids = df[
    df["record_id"].duplicated(keep=False)
].sort_values("record_id")

print("Rows with repeated IDs:", len(duplicate_ids))

print("\nRepeated record IDs:")
print(
    duplicate_ids["record_id"].value_counts().to_string()
)

# Check whether duplicate IDs have identical rows
print("\n========== OUTLIER INVESTIGATION ==========")

Q1 = df["waste_kg"].quantile(0.25)
Q3 = df["waste_kg"].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Upper outlier limit:", upper_limit)

outliers = df[df["waste_kg"] > upper_limit]

print("Potential outlier count:", len(outliers))

print(
    outliers[["record_id", "waste_kg"]]
    .to_string(index=False)
)


print("\n========== EXACT DUPLICATE CHECK ==========")

print("Exact duplicate rows:", df.duplicated().sum())

print("\n========== DUPLICATE ID ANALYSIS ==========")

duplicate_ids = df[
    df["record_id"].duplicated(keep=False)
].sort_values("record_id")

for record_id, group in duplicate_ids.groupby("record_id"):

    print(f"\nRecord ID: {record_id}")
    print("Occurrences:", len(group))

    print("Unique complete rows:", len(group.drop_duplicates()))

    print("Unique target values:", group["waste_kg"].nunique())

print("\n========== OUTLIER DETAILS ==========")

outliers = df[df["waste_kg"] > 79.7]

print("Outlier count:", len(outliers))

print(outliers.to_string(index=False))

from src.mp270.preprocessing import clean_data

cleaned_df = clean_data(df)

print("\n========== CLEANING RESULTS ==========")

print("Before cleaning:", len(df))
print("After cleaning:", len(cleaned_df))

print("Remaining exact duplicates:",
      cleaned_df.duplicated().sum())
