
import os
import joblib
import pandas as pd

from src.mp270.preprocessing import prepare_features


# =====================================
# 1. LOAD CHAMPION MODEL (V2)
# =====================================

model = joblib.load(
    "models/model.joblib"
)

print("Champion model loaded successfully!")


# =====================================
# 2. LOAD HOLDOUT FEATURES
# =====================================

df = pd.read_csv(
    "data/raw/holdout_features.csv"
)

print("Holdout dataset shape:", df.shape)


# =====================================
# 3. PREPARE FEATURES
# =====================================

X_holdout = prepare_features(df)

print("Holdout feature shape:", X_holdout.shape)


# =====================================
# 4. GENERATE PREDICTIONS
# =====================================

predictions = model.predict(X_holdout)


# =====================================
# 5. CREATE REQUIRED OUTPUT
# =====================================

output = pd.DataFrame({
    "record_id": df["record_id"],
    "prediction": predictions
})

os.makedirs("predictions", exist_ok=True)

output.to_csv(
    "predictions/holdout_predictions.csv",
    index=False
)

print("\nHoldout predictions generated!")
print("Total predictions:", len(output))
print("Output columns:", list(output.columns))
print("Saved to: predictions/holdout_predictions.csv")

print("\nFirst 5 predictions:")
print(output.head())

