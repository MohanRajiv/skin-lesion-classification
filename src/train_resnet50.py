import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from resnet50_model import NUM_CLASSES, build_resnet50, resnet50_transforms
from train_ham10000 import CLASSES, load_splits


# set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SkinDataset(Dataset):
    def __init__(self, df, training=False):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = resnet50_transforms(training=training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        x = self.transform(img)
        return x, self.labels[idx]


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)
            
    return y_true, y_pred


def main(args):
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "best_resnet50.pt"

    train_df, val_df, test_df = load_splits(args.data_dir, split_dir=args.split_dir)
    print(f"Using {device} | train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_s = train_df[["image_id", "lesion_id", "dx"]].copy()
    train_s["split"] = "train"
    val_s = val_df[["image_id", "lesion_id", "dx"]].copy()
    val_s["split"] = "validation"
    test_s = test_df[["image_id", "lesion_id", "dx"]].copy()
    test_s["split"] = "test"
    
    splits_df = pd.concat([train_s, val_s, test_s], ignore_index=True)
    splits_df.to_csv(args.output_dir / "splits.csv", index=False)

    train_ds = SkinDataset(train_df, training=True)
    val_ds = SkinDataset(val_df, training=False)
    test_ds = SkinDataset(test_df, training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_resnet50(pretrained=True)
    model = model.to(device)

    # setup optimizer and freeze layers if not fine-tuning everything
    if args.fine_tune_all:
        backbone_params = []
        for n, p in model.named_parameters():
            if not n.startswith(("layer4", "fc")):
                backbone_params.append(p)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
    else:
        for name, param in model.named_parameters():
            if name.startswith(("layer4", "fc")):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)

    counts = np.bincount(train_df["label"], minlength=NUM_CLASSES)
    weights = len(train_df) / (NUM_CLASSES * counts)
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0.0
    history = []

    # training loop
    for epoch in range(1, args.epochs + 1):
        if args.fine_tune_all:
            model.train()
        else:
            model.eval()
            model.layer4.train()
            model.fc.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # validate
        y_true, y_pred = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        history.append({
            'epoch': epoch,
            'validation_accuracy': val_acc,
            'validation_macro_f1': val_f1
        })
        print(f"Epoch {epoch}/{args.epochs} - val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f}")

        # save best model based on macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_macro_f1': best_f1,
                'classes': CLASSES
            }, ckpt_path)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(args.output_dir / "validation_metrics.csv", index=False)

    # test set
    if args.evaluate_test:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        y_true, y_pred = evaluate(model, test_loader, device)

        acc = accuracy_score(y_true, y_pred)
        print(f"\nBest epoch: {ckpt['epoch']} | Test acc: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0))

        rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
        pd.DataFrame(rep).transpose().to_csv(args.output_dir / "test_classification_report.csv")

        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(args.output_dir / "test_confusion_matrix.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet50"))
    parser.add_argument("--fine-tune-all", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    
    args = parser.parse_args()
    main(args)