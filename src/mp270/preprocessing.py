
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def normalize_categories(df):
    df = df.copy()

    if "meal" in df.columns:
        df["meal"] = (
            df["meal"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({
                "bre.": "breakfast",
                "lun.": "lunch",
                "din.": "dinner",
            })
        )

    return df

NUMERIC_FEATURES = [
    "students_registered",
    "attendance_pct",
    "menu_rating",
    "temp_c",
    "cooked_kg",
    "plate_size_cm",
    "date_year",
    "date_month",
    "date_day",
    "date_dayofweek",
]

CATEGORICAL_FEATURES = [
    "meal",
    "hostel",
]


def clean_data(df):
    """Remove exact duplicate records."""

    df = df.copy()

    print("Rows before cleaning:", len(df))

    df = df.drop_duplicates().copy()

    print("Rows after cleaning:", len(df))

    if "record_id" in df.columns:
        print(
            "Duplicate record IDs:",
            df["record_id"].duplicated().sum()
        )

        df = normalize_categories(df)

    return df


def prepare_features(df):
    """Prepare common features for training and prediction."""

    df = df.copy()

    # Convert date into datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Extract date features
    df["date_year"] = df["date"].dt.year
    df["date_month"] = df["date"].dt.month
    df["date_day"] = df["date"].dt.day
    df["date_dayofweek"] = df["date"].dt.day_of_week

    # Remove identifier and original date
    df = df.drop(
        columns=["record_id", "date"],
        errors="ignore"
    )

    # Exclude target and unavailable measurement
    df = df.drop(
        columns=["waste_kg", "waste_weighed_at_bin"],
        errors="ignore"
    )

    return df


def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor
