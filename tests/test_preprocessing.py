
import pandas as pd

from src.mp270.preprocessing import (
    clean_data,
    prepare_features,
    build_preprocessor,
)


def test_preprocessing():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "students_registered": [100, 110, 120],
        "attendance_pct": [80, 85, 90],
        "menu_rating": [3.5, 4.0, 4.5],
        "temp_c": [25, 26, 27],
        "cooked_kg": [100, 110, 120],
        "plate_size_cm": [25, 26, 27],
        "meal": ["bre.", "lun.", "din."],
        "hostel": ["Girls Hostel", "Boys Hostel 1", "Boys Hostel 2"],
        "waste_kg": [10, 12, 15],
        "waste_weighed_at_bin": [11, 13, 16],
    })

    df = clean_data(df)

    y = df["waste_kg"]
    X = prepare_features(df)

    assert "waste_kg" not in X.columns
    assert "waste_weighed_at_bin" not in X.columns

    assert set(X["meal"]) == {"breakfast", "lunch", "dinner"}

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    assert X_processed.shape[0] == 3
    assert y.shape[0] == 3


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "meal": ["lunch", "lunch"],
    })

    cleaned = clean_data(df)

    assert len(cleaned) == 1


def test_prepare_features_creates_date_features():
    df = pd.DataFrame({
        "record_id": ["R1"],
        "date": ["2026-01-01"],
        "students_registered": [100],
        "attendance_pct": [80],
        "menu_rating": [4.0],
        "temp_c": [25],
        "cooked_kg": [100],
        "plate_size_cm": [25],
        "meal": ["lunch"],
        "hostel": ["Girls Hostel"],
        "waste_kg": [10],
        "waste_weighed_at_bin": [11],
    })

    X = prepare_features(df)

    assert X.loc[0, "date_year"] == 2026
    assert X.loc[0, "date_month"] == 1
    assert X.loc[0, "date_day"] == 1
    assert "date_dayofweek" in X.columns
