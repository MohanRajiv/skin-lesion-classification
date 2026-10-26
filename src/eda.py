from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

metadata_path = Path("data/raw/HAM10000_metadata.csv")
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metadata_path)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Missing age values: {df['age'].isna().sum()}")

class_order = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
class_counts = df["dx"].value_counts().reindex(class_order, fill_value=0)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax, color="steelblue")
ax.set(title="HAM10000 images by diagnosis", xlabel="Diagnosis code", ylabel="Image count")
fig.tight_layout()
fig.savefig(output_dir / "class_distribution.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df["age"].dropna(), bins=25, ax=ax, color="darkorange")
ax.set(title="HAM10000 age distribution", xlabel="Age", ylabel="Image count")
fig.tight_layout()
fig.savefig(output_dir / "age_distribution.png", dpi=200)
plt.close(fig)

print(f"Saved figures in {output_dir}")
print("\nClass counts:")
print(class_counts)

