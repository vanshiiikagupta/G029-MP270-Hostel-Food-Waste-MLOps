
import pandas as pd
import numpy as np

# Load later predictions
pred = pd.read_csv("reports/later_predictions.csv")

# Load original later dataset
later = pd.read_csv("data/raw/later.csv")

# Join predictions with hostel and input information
df = later.merge(
    pred[["record_id", "predicted_waste_kg", "absolute_error"]],
    on="record_id",
    how="inner"
)

# Calculate signed error
df["signed_error"] = (
    df["predicted_waste_kg"] - df["waste_kg"]
)

print("\n========== ERROR ANALYSIS BY HOSTEL ==========")

summary = df.groupby("hostel").agg(
    records=("record_id", "count"),
    actual_mean=("waste_kg", "mean"),
    predicted_mean=("predicted_waste_kg", "mean"),
    mean_absolute_error=("absolute_error", "mean"),
    mean_signed_error=("signed_error", "mean"),
    cooked_mean=("cooked_kg", "mean")
)

print(summary.round(3))

print("\n========== BOYS HOSTEL 2 ERROR DETAILS ==========")

boys2 = df[df["hostel"] == "Boys Hostel 2"].copy()

print(
    boys2[
        [
            "record_id",
            "date",
            "meal",
            "cooked_kg",
            "waste_kg",
            "predicted_waste_kg",
            "absolute_error"
        ]
    ].sort_values("absolute_error", ascending=False).head(15).to_string(index=False)
)

print("\n========== SAVING ERROR REPORT ==========")

summary.to_csv("reports/hostel_error_analysis.csv")

print("Saved: reports/hostel_error_analysis.csv")
