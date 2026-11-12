import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from densenet121_model import NUM_CLASSES, build_densenet121, densenet121_transforms
from train_ham10000 import CLASSES, load_splits


SEED = 42
BATCH_SIZE = 16
TRAINABLE_FEATURES = {"denseblock4", "norm5"}


class DenseNetSkinDataset(Dataset):
    def __init__(self, frame, training=False):
        self.paths = frame['path'].tolist()
        self.labels = frame['label'].tolist()
        self.transform = densenet121_transforms(training=training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        x = self.transform(img)
        return x, self.labels[index]


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    return actual, predicted


def main(args):
    # set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_densenet121.pt"

    train_df, val_df, test_df = load_splits(args.data_dir, split_dir=args.split_dir)
    print(f"Device: {device} | Images: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    pd.concat([
        frame[["image_id", "lesion_id", "dx"]].assign(split=name)
        for name, frame in zip(("train", "validation", "test"), (train_df, val_df, test_df))
    ], ignore_index=True).to_csv(output_dir / "splits.csv", index=False)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        DenseNetSkinDataset(train_df, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(DenseNetSkinDataset(val_df), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(DenseNetSkinDataset(test_df), batch_size=args.batch_size, num_workers=0)

    model = build_densenet121(pretrained=True).to(device)

    # freeze earlier layers if not fine-tuning the full network
    if not args.fine_tune_all:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, module in model.features.named_children():
            if name in TRAINABLE_FEATURES:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    class_counts = np.bincount(train_df["label"], minlength=NUM_CLASSES)
    class_weights = len(train_df) / (NUM_CLASSES * class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    if args.fine_tune_all:
        optimizer = torch.optim.AdamW([
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-3},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.features.denseblock4.parameters(), 'lr': 1e-4},
            {'params': model.features.norm5.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3},
        ], weight_decay=1e-4)

    best_macro_f1 = -1.0
    epoch_metrics = []

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.fine_tune_all:
            for name, module in model.features.named_children():
                if name not in TRAINABLE_FEATURES:
                    module.eval()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        # validate
        actual, predicted = predict(model, val_loader, device)
        val_accuracy = accuracy_score(actual, predicted)
        val_macro_f1 = f1_score(
            actual, predicted, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0
        )
        epoch_metrics.append({
            'epoch': epoch,
            'validation_accuracy': val_accuracy,
            'validation_macro_f1': val_macro_f1,
        })
        print(f"Epoch {epoch:02d}/{args.epochs} | val accuracy={val_accuracy:.3f} | val macro-F1={val_macro_f1:.3f}")

        # save best model based on macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({
                'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'classes': CLASSES,
                'num_classes': NUM_CLASSES,
                'seed': args.seed,
                'epoch': epoch,
                'best_val_macro_f1': float(best_macro_f1),
                'fine_tune_all': args.fine_tune_all,
            }, checkpoint_path)

    pd.DataFrame(epoch_metrics).to_csv(output_dir / "validation_metrics.csv", index=False)
    print(f"Saved best validation checkpoint: {checkpoint_path}")

    # test set
    if args.evaluate_test:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        actual, predicted = predict(model, test_loader, device)

        print(f"\nSelected epoch: {checkpoint['epoch']}")
        print(f"Test accuracy: {accuracy_score(actual, predicted):.3f}")
        print(classification_report(
            actual, predicted, labels=list(range(NUM_CLASSES)), target_names=CLASSES, digits=3, zero_division=0
        ))

        report = classification_report(
            actual, predicted, labels=list(range(NUM_CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(output_dir / "test_classification_report.csv")

        matrix = confusion_matrix(actual, predicted, labels=list(range(NUM_CLASSES)))
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(output_dir / "test_confusion_matrix.csv")
    else:
        print("Skipped test evaluation. Add --evaluate-test for the final report.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/densenet121"))
    parser.add_argument("--fine-tune-all", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")

    args = parser.parse_args()
    main(args)