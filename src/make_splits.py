from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/raw/HAM10000_metadata.csv")

if df["lesion_id"].isna().any():
    raise ValueError("Some rows are missing lesion_id.")

if (df.groupby("lesion_id")["dx"].nunique() > 1).any():
    raise ValueError("At least one lesion_id has multiple diagnosis labels.")

# Hold out about 1/7 of the images for testing, keeping lesions together.
outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
trainval_idx, test_idx = next(
    outer.split(df, y=df["dx"], groups=df["lesion_id"])
)
trainval = df.iloc[trainval_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)

# Split the remainder into training and validation sets.
inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=43)
train_idx, val_idx = next(
    inner.split(trainval, y=trainval["dx"], groups=trainval["lesion_id"])
)
train = trainval.iloc[train_idx].reset_index(drop=True)
val = trainval.iloc[val_idx].reset_index(drop=True)

splits = {"train": train, "val": val, "test": test}

# Confirm that no lesion appears in multiple splits.
for first_name, first in splits.items():
    for second_name, second in splits.items():
        if first_name < second_name:
            overlap = set(first["lesion_id"]) & set(second["lesion_id"])
            if overlap:
                raise ValueError(
                    f"Lesions appear in both {first_name} and {second_name}."
                )

out_dir = Path("data/splits")
out_dir.mkdir(parents=True, exist_ok=True)
classes = sorted(df["dx"].unique())

for name, split in splits.items():
    split.to_csv(out_dir / f"{name}.csv", index=False)
    counts = split["dx"].value_counts().reindex(classes, fill_value=0)
    percentages = (counts / len(split) * 100).round(2)

    print(f"\n{name}: {len(split)} images, {split['lesion_id'].nunique()} lesions")
    print(pd.DataFrame({"images": counts, "percent": percentages}))

print(f"\nSaved split files in {out_dir}")
print("No lesion_id appears in more than one split.")
