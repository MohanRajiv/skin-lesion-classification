"""Train a small, seven-class CNN on the original HAM10000 JPEG images.

Install:
    python -m pip install torch torchvision pandas numpy scikit-learn pillow
Run:
    python train_ham10000.py /path/to/HAM10000 --epochs 10

The folder must contain HAM10000_metadata.csv and the extracted image folders.
Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Paper: https://doi.org/10.1038/sdata.2018.161
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def load_splits(data_dir, split_dir="data/splits"):
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    metadata = pd.read_csv(data_dir / "HAM10000_metadata.csv")

    image_paths = {
        path.stem: str(path)
        for path in sorted(data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    label_map = {name: i for i, name in enumerate(CLASSES)}
    frames = {}

    for name in ("train", "val", "test"):
        frame = pd.read_csv(split_dir / f"{name}.csv")

        frame["path"] = frame["image_id"].astype(str).map(image_paths)
        if frame["path"].isna().any():
            missing = frame.loc[frame["path"].isna(), "image_id"].head().tolist()
            raise FileNotFoundError(f"Missing image files for: {missing}")

        frame["label"] = frame["dx"].map(label_map)
        if frame["label"].isna().any():
            raise ValueError(f"{name}.csv contains an unknown diagnosis code.")

        frames[name] = frame.reset_index(drop=True)

    for i, first_name in enumerate(frames):
        for second_name in list(frames)[i + 1:]:
            overlap = (
                set(frames[first_name]["lesion_id"])
                & set(frames[second_name]["lesion_id"])
            )
            if overlap:
                raise ValueError(
                    f"Lesions appear in both {first_name} and {second_name}."
                )

    return frames["train"], frames["val"], frames["test"]

class SkinDataset(Dataset):
    def __init__(self, frame, training=False):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].tolist()
        steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if training:
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.labels[index]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, len(CLASSES)),  # Raw logits; no softmax before CrossEntropyLoss.
        )

    def forward(self, images):
        return self.layers(images)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        actual.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return actual, predicted


def main(data_dir, split_dir="data/splits", epochs=20, output_dir="outputs/baseline_cnn", evaluate_test=False,):    
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_ham10000_cnn.pt"

    train_df, val_df, test_df = load_splits(data_dir, split_dir=split_dir)
    print(f"Images: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    # Preserve the exact split for later comparison and reproducibility.
    pd.concat([
        frame[["image_id", "lesion_id", "dx"]].assign(split=name)
        for name, frame in zip(("train", "validation", "test"), (train_df, val_df, test_df))
    ]).to_csv(output_dir / "splits.csv", index=False)

    train_loader = DataLoader(
        SkinDataset(train_df, training=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(SkinDataset(val_df), batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(SkinDataset(test_df), batch_size=BATCH_SIZE, num_workers=0)

    # Estimate class weights using ONLY training labels.
    counts = np.bincount(train_df["label"], minlength=len(CLASSES))
    if (counts == 0).any():
        raise ValueError("Every class must be represented in the training split.")
    weights = torch.tensor(len(train_df) / (len(CLASSES) * counts), dtype=torch.float32)
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        actual, predicted = evaluate(model, val_loader, device)
        val_acc = accuracy_score(actual, predicted)
        val_f1 = f1_score(
            actual, predicted, labels=list(range(len(CLASSES))), average="macro", zero_division=0
        )
        print(f"Epoch {epoch:02d}/{epochs} | val accuracy={val_acc:.3f} | val macro-F1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
                "best_val_macro_f1": float(best_f1),
            }, checkpoint_path)

    if evaluate_test:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        actual, predicted = evaluate(model, test_loader, device)
        print(f"\nTest accuracy: {accuracy_score(actual, predicted):.3f}")
        print(classification_report(
            actual,
            predicted,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            digits=3,
            zero_division=0,
        ))
    else:
        print("\nSkipped test evaluation. Use --evaluate-test for the final evaluation.")

    print(f"Saved model: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Folder containing HAM10000_metadata.csv and JPEGs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("ham10000_output"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    main(
        args.data_dir,
        split_dir=args.split_dir,
        epochs=args.epochs,
        output_dir=args.output_dir,
        evaluate_test=args.evaluate_test,
    )