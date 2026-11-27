
import os
import hashlib
import joblib
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.mp270.preprocessing import (
    prepare_features,
    build_preprocessor
)


# =====================================
# 1. LOAD CONFIGURATION
# =====================================

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

RANDOM_STATE = params["random_state"]
TEST_SIZE = params["test_size"]

# Original raw file is used only for its SHA-256 fingerprint.
RAW_DATA_PATH = "data/raw/train.csv"
DATA_PATH = "data/processed/train_cleaned.csv"

def calculate_sha256(filepath):
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


data_sha256 = calculate_sha256(RAW_DATA_PATH)

# =====================================
# 2. LOAD CORRECTED DATA
# =====================================

df = pd.read_csv(DATA_PATH)

print("Corrected dataset shape:", df.shape)
print("Original data SHA-256:", data_sha256)

# Separate target
y = df["waste_kg"]

# Prepare features
# prepare_features must exclude waste_kg and waste_weighed_at_bin.
X = prepare_features(df)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# =====================================
# 3. TRAIN VALIDATION SPLIT
# =====================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))

# =====================================
# 4. DEFINE MODELS FROM params.yaml
# =====================================

model_params = params["models"]

models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=model_params["decision_tree"]["max_depth"],
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=model_params["random_forest"]["n_estimators"],
        max_depth=model_params["random_forest"]["max_depth"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=model_params["gradient_boosting"]["n_estimators"],
        learning_rate=model_params["gradient_boosting"]["learning_rate"],
        max_depth=model_params["gradient_boosting"]["max_depth"],
        random_state=RANDOM_STATE
    )
}

# =====================================
# 5. MLFLOW SETUP
# =====================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "MP270-Hostel-Food-Waste"
REGISTERED_MODEL_NAME = "MP270-Hostel-Food-Waste-Model"

mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

results = []
best_mae = float("inf")
best_model = None
best_model_name = None
best_run_id = None

# =====================================
# 6. TRAIN AND EVALUATE
# =====================================

for name, estimator in models.items():

    print("\n" + "=" * 50)
    print("TRAINING:", name)
    print("=" * 50)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator)
        ]
    )

    with mlflow.start_run(run_name=name) as run:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        r2 = r2_score(y_val, predictions)

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))

        # Log required parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("data_sha256", data_sha256)

        # Log model-specific parameters
        mlflow.log_params(estimator.get_params())

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_model_name = name
            best_run_id = run.info.run_id


# =====================================
# 7. DISPLAY RESULTS
# =====================================

results_df = pd.DataFrame(results).sort_values(by="MAE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBEST MODEL:", best_model_name)
print("BEST MAE:", best_mae)

# =====================================
# 8. ASSIGN CHAMPION ALIAS
# =====================================

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

best_version = None

for version in versions:
    if version.run_id == best_run_id:
        best_version = version.version
        break

if best_version is None:
    raise RuntimeError("Could not find the registered best model version.")

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=best_version
)

print("Champion alias assigned to model version:", best_version)

# =====================================
# 9. SAVE BEST MODEL AND REPORT
# =====================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    best_model,
    "models/model.joblib"
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nChampion model saved successfully!")
print("Location: models/model.joblib")
