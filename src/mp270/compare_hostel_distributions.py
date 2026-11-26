
import pandas as pd

train = pd.read_csv("data/processed/train_cleaned.csv")
later = pd.read_csv("data/raw/later.csv")

for hostel in sorted(later["hostel"].unique()):

    train_group = train[train["hostel"] == hostel]
    later_group = later[later["hostel"] == hostel]

    print("\n" + "=" * 65)
    print("HOSTEL:", hostel)
    print("=" * 65)

    print("Training records:", len(train_group))
    print("Later records:", len(later_group))

    for column in ["cooked_kg", "waste_kg", "students_registered"]:

        print(f"\n{column}")

        print(
            "Training mean:",
            round(train_group[column].mean(), 2)
        )

        print(
            "Later mean:",
            round(later_group[column].mean(), 2)
        )

        print(
            "Training max:",
            round(train_group[column].max(), 2)
        )

        print(
            "Later max:",
            round(later_group[column].max(), 2)
        )
