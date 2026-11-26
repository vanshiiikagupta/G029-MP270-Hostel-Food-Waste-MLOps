
import os
import pandas as pd

# Load original training data and remove exact duplicates
df = pd.read_csv("data/raw/train.csv")
df = df.drop_duplicates().copy()

# Calculate ratio between recorded waste and bin measurement
df["waste_bin_ratio"] = (
    df["waste_kg"] / df["waste_weighed_at_bin"]
)

# Identify possible scaling errors
scaling_candidates = df[
    df["waste_bin_ratio"].between(9.5, 10.5)
].copy()

# Identify other unusual ratios for separate investigation
other_candidates = df[
    (
        (df["waste_bin_ratio"] > 1.1) |
        (df["waste_bin_ratio"] < 0.9)
    ) &
    (~df.index.isin(scaling_candidates.index))
].copy()

# Label findings
scaling_candidates["audit_finding"] = (
    "Possible 10x scaling error - verify"
)

other_candidates["audit_finding"] = (
    "Other unusual ratio - investigate"
)

# Combine findings
audit_report = pd.concat(
    [scaling_candidates, other_candidates]
)

# Save report
os.makedirs("reports", exist_ok=True)

output_path = "reports/data_quality_audit.csv"
audit_report.to_csv(output_path, index=False)

print("DATA QUALITY AUDIT COMPLETED")
print("Cleaned training rows:", len(df))
print("Possible 10x scaling errors:", len(scaling_candidates))
print("Other unusual ratios:", len(other_candidates))
print("Total flagged records:", len(audit_report))
print("Report saved to:", output_path)

print("\nFlagged records:")
print(
    audit_report[
        ["record_id", "waste_kg",
         "waste_weighed_at_bin",
         "waste_bin_ratio", "audit_finding"]
    ].to_string(index=False)
)
