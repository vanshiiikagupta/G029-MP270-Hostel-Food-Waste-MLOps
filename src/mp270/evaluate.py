
import os
import joblib
import yaml
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from src.mp270.preprocessing import (
    clean_data,
    prepare_features
)


# =================================
# 1. LOAD CONFIGURATION AND MODEL
# =================================

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

model = joblib.load("models/champion_model.joblib")

print("Champion model loaded successfully!")

# =================================
# 2. LOAD CORRECTED TRAINING DATA
# =================================

train_df = pd.read_csv("data/processed/train_cleaned.csv")
train_df = clean_data(train_df)

X_train_reference = prepare_features(train_df)

# =================================
# 3. LOAD LATER DATASET
# =================================

later_df = pd.read_csv("data/raw/later.csv")
print("\nOriginal later dataset:", later_df.shape)

later_df = clean_data(later_df)

y_actual = later_df["waste_kg"]
X_later = prepare_features(later_df)

print("Cleaned later dataset:", later_df.shape)
print("Evaluation features:", X_later.shape)

# =================================
# 4. CALCULATE PSI FOR NUMERIC FEATURES
# =================================

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index using
    quantile bins derived from the reference data.
    """

    reference = pd.to_numeric(
        reference, errors="coerce"
    )
    current = pd.to_numeric(
        current, errors="coerce"
    )

    reference_values = reference.dropna()

    if reference_values.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)

    internal_edges = np.unique(
        np.nanquantile(reference_values, quantiles)
    )

    if len(internal_edges) == 1:
        value = internal_edges[0]
        epsilon = max(abs(value) * 0.000001, 0.000001)

        edges = [
            -np.inf,
            value - epsilon,
            value + epsilon,
            np.inf
        ]
    else:
        edges = np.concatenate((
            [-np.inf],
            internal_edges[1:-1],
            [np.inf]
        ))

    reference_binned = pd.cut(
        reference, bins=edges, include_lowest=True
    ).astype(object).fillna("MISSING")

    current_binned = pd.cut(
        current, bins=edges, include_lowest=True
    ).astype(object).fillna("MISSING")

    categories = list(reference_binned.unique())

    for category in current_binned.unique():
        if category not in categories:
            categories.append(category)

    reference_pct = (
        reference_binned.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    current_pct = (
        current_binned.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    epsilon = 0.0001

    reference_pct = reference_pct.clip(lower=epsilon)
    current_pct = current_pct.clip(lower=epsilon)

    psi = (
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    ).sum()

    return float(psi)


numeric_columns = X_train_reference.select_dtypes(
    include=np.number
).columns.tolist()

psi_results = []

for column in numeric_columns:

    psi_value = calculate_psi(
        X_train_reference[column],
        X_later[column]
    )

    psi_results.append({
        "feature": column,
        "PSI": psi_value
    })

psi_df = pd.DataFrame(psi_results).sort_values(
    by="PSI", ascending=False
)

print("\n========== PSI RESULTS ==========")
print(psi_df.to_string(index=False))

os.makedirs("reports", exist_ok=True)

psi_df.to_csv(
    "reports/psi_report.csv",
    index=False
)

print("\nPSI report saved: reports/psi_report.csv")

# =================================
# 5. GENERATE V1 PREDICTIONS
# =================================

predictions = model.predict(X_later)

# =================================
# 6. OVERALL EVALUATION
# =================================

mae = mean_absolute_error(y_actual, predictions)

rmse = np.sqrt(
    mean_squared_error(y_actual, predictions)
)

r2 = r2_score(y_actual, predictions)

print("\n========== V1 LATER EVALUATION ==========")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

# =================================
# 7. BASELINE COMPARISON
# =================================

y_train = train_df["waste_kg"]

_, y_train_validation = train_test_split(
    y_train,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

baseline_prediction = y_train_validation.mean()

baseline_predictions = np.full(
    len(y_actual),
    baseline_prediction
)

baseline_mae = mean_absolute_error(
    y_actual, baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")
print("Baseline MAE:", round(baseline_mae, 4))
print("Model MAE:", round(mae, 4))

# =================================
# 8. MAE BY HOSTEL
# =================================

hostel_results = []

for hostel, group in later_df.assign(
    prediction=predictions
).groupby("hostel"):

    hostel_mae = mean_absolute_error(
        group["waste_kg"],
        group["prediction"]
    )

    hostel_results.append({
        "hostel": hostel,
        "records": len(group),
        "MAE": hostel_mae
    })

hostel_df = pd.DataFrame(hostel_results)

print("\n========== MAE BY HOSTEL ==========")
print(hostel_df.to_string(index=False))

hostel_df.to_csv(
    "reports/mae_by_hostel.csv",
    index=False
)

# =================================
# 9. SAVE LATER PREDICTIONS
# =================================

results = pd.DataFrame({
    "record_id": later_df["record_id"].values,
    "actual_waste_kg": y_actual.values,
    "predicted_waste_kg": predictions,
    "absolute_error": np.abs(
        y_actual.values - predictions
    )
})

results.to_csv(
    "reports/later_predictions.csv",
    index=False
)

print("\nSaved reports:")
print("- reports/psi_report.csv")
print("- reports/mae_by_hostel.csv")
print("- reports/later_predictions.csv")
