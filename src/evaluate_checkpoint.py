from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from train_ham10000 import (
    BATCH_SIZE,
    CLASSES,
    SimpleCNN,
    SkinDataset,
    evaluate,
    load_splits,
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_, _, test_df = load_splits("data/raw", "data/splits")
test_loader = DataLoader(
    SkinDataset(test_df),
    batch_size=BATCH_SIZE,
    num_workers=0,
)

checkpoint_path = Path("outputs/baseline_cnn/best_ham10000_cnn.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

model = SimpleCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])

actual, predicted = evaluate(model, test_loader, device)

print(f"Test accuracy: {accuracy_score(actual, predicted):.3f}")
print(classification_report(
    actual,
    predicted,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=3,
    zero_division=0,
))
