
import os
from pathlib import Path
import hashlib
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.mp270.preprocessing import (
    clean_data,
    prepare_features,
    build_preprocessor
)


# =========================================
# 1. LOAD CONFIGURATION
# =========================================

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

random_state = params["random_state"]

# =========================================
# 2. LOAD ORIGINAL CORRECTED TRAINING DATA
# =========================================

train_df = pd.read_csv(
    "data/processed/train_cleaned.csv"
)

train_df = clean_data(train_df)

print("Original corrected training rows:", len(train_df))


# =========================================
# 3. LOAD AND SORT LATER DATA
# =========================================

later_df = pd.read_csv("data/raw/later.csv")

later_df["date"] = pd.to_datetime(
    later_df["date"],
    errors="coerce"
)

later_df = later_df.sort_values(
    "date"
).reset_index(drop=True)

later_df = clean_data(later_df)

split_index = int(len(later_df) * 0.70)

later_train = later_df.iloc[:split_index].copy()
later_test = later_df.iloc[split_index:].copy()

print("Later training rows:", len(later_train))
print("Final evaluation rows:", len(later_test))


# =========================================
# 4. COMBINE TRAINING DATA
# =========================================

combined_train = pd.concat(
    [train_df, later_train],
    ignore_index=True
)

combined_train = clean_data(combined_train)

print("Combined v2 training rows:", len(combined_train))


# =========================================
# 5. PREPARE FEATURES
# =========================================

X_train = prepare_features(combined_train)
y_train = combined_train["waste_kg"]

X_test = prepare_features(later_test)
y_test = later_test["waste_kg"]


# =========================================
# 6. LOAD V1 MODEL
# =========================================

v1_model = joblib.load(
    "models/v1_model.joblib"
)

v1_predictions = v1_model.predict(X_test)

v1_mae = mean_absolute_error(
    y_test,
    v1_predictions
)

print("\nV1 MAE:", round(v1_mae, 4))


# =========================================
# 7. DEFINE V2 MODEL
# =========================================

v2_model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=params["models"]["gradient_boosting"]["n_estimators"],
                learning_rate=params["models"]["gradient_boosting"]["learning_rate"],
                max_depth=params["models"]["gradient_boosting"]["max_depth"],
                random_state=random_state
            )
        )
    ]
)


# =========================================
# 8. TRAIN V2
# =========================================

v2_model.fit(
    X_train,
    y_train
)

v2_predictions = v2_model.predict(X_test)

v2_mae = mean_absolute_error(
    y_test,
    v2_predictions
)

print("V2 MAE:", round(v2_mae, 4))


# =========================================
# 9. CHECK PROMOTION RULE
# =========================================

improvement_pct = (
    (v1_mae - v2_mae) / v1_mae
) * 100

promote_v2 = improvement_pct >= 5

print("\n========== V1 VS V2 ==========")
print("V1 MAE:", round(v1_mae, 4))
print("V2 MAE:", round(v2_mae, 4))
print("Improvement:", round(improvement_pct, 2), "%")

if promote_v2:
    print("DECISION: PROMOTE V2")
else:
    print("DECISION: KEEP V1")


# =========================================
# 10. MLFLOW LOGGING
# =========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Person 3 uses a separate experiment and local artifact folder
experiment_name = "MP270-Hostel-Food-Waste-Person3"
artifact_root = Path("mlartifacts/person3").resolve()

os.makedirs(artifact_root, exist_ok=True)

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_root.as_uri()
    )
else:
    if not experiment.artifact_location.startswith("file:"):
        raise RuntimeError(
            "Person 3 MLflow experiment has an incompatible artifact location."
        )

mlflow.set_experiment(experiment_name)

# Fingerprint of original raw training CSV
sha256 = hashlib.sha256()

with open("data/raw/train.csv", "rb") as file:
    for chunk in iter(lambda: file.read(4096), b""):
        sha256.update(chunk)

data_sha256 = sha256.hexdigest()

registered_model_name = "MP270-Hostel-Food-Waste-Model"

with mlflow.start_run(run_name="V2-Retraining"):

    mlflow.log_param("model_version", "v2")
    mlflow.log_param("training_rows", len(combined_train))
    mlflow.log_param("later_training_rows", len(later_train))
    mlflow.log_param("evaluation_rows", len(later_test))
    mlflow.log_param("data_sha256", data_sha256)

    mlflow.log_metric("v1_MAE", v1_mae)
    mlflow.log_metric("v2_MAE", v2_mae)
    mlflow.log_metric("improvement_percent", improvement_pct)

    mlflow.sklearn.log_model(
        sk_model=v2_model,
        name="model",
        serialization_format="cloudpickle"
    )


# =========================================
# 11. SAVE COMPARISON REPORT
# =========================================

os.makedirs("reports", exist_ok=True)

comparison = pd.DataFrame([
    {
        "model": "v1",
        "MAE": v1_mae
    },
    {
        "model": "v2",
        "MAE": v2_mae
    }
])

comparison.to_csv(
    "reports/v1_v2_comparison.csv",
    index=False
)

# Save predictions for audit
prediction_report = pd.DataFrame({
    "record_id": later_test["record_id"].values,
    "actual_waste_kg": y_test.values,
    "v1_prediction": v1_predictions,
    "v2_prediction": v2_predictions
})

prediction_report.to_csv(
    "reports/v1_v2_predictions.csv",
    index=False
)


# =========================================
# 12. PROMOTE V2 IF IT PASSES
# =========================================

if promote_v2:

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run(run_name="V2-Registration"):
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("training_rows", len(combined_train))
        mlflow.log_param("later_training_rows", len(later_train))
        mlflow.log_param("evaluation_rows", len(later_test))
        mlflow.log_param("data_sha256", data_sha256)

        mlflow.log_metric("v1_MAE", v1_mae)
        mlflow.log_metric("v2_MAE", v2_mae)
        mlflow.log_metric("improvement_percent", improvement_pct)

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=registered_model_name
        )

    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=model_info.registered_model_version
    )

    joblib.dump(
        v2_model,
        "models/model.joblib"
    )

    print("\nV2 promoted and saved as champion.")

else:
    print("\nExisting champion remains unchanged. V2 was not promoted.")


print("\nReports saved successfully.")
