import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision import datasets, transforms

from models.cnn import CNN
from train import build_dataloaders, FASHION_MEAN, FASHION_STD

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def plot_history(results_dir: Path):
    p = results_dir / "history.json"
    if not p.exists():
        return
    h = json.loads(p.read_text())
    ep = h["epoch"]

    plt.figure()
    plt.plot(ep, h["train_loss"], label="train")
    plt.plot(ep, h["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.savefig(results_dir / "loss.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(ep, h["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Validation accuracy")
    plt.savefig(results_dir / "accuracy.png", dpi=150); plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_dataloaders(args.data_dir, args.batch_size, seed=args.seed)

    model = CNN().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.append(preds.cpu()); all_targets.append(y.cpu())

    test_loss = total_loss / total
    test_acc = correct / total
    print(f"TEST | loss={test_loss:.4f} | accuracy={test_acc:.4f}")

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.tight_layout(); plt.savefig(results_dir / "confusion_matrix.png", dpi=150); plt.close()

    # примеры предсказаний
    ds = test_loader.dataset
    idxs = np.random.RandomState(args.seed).choice(len(ds), 8, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, i in zip(axes.flat, idxs):
        img, label = ds[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"true: {CLASS_NAMES[label]}\npred: {CLASS_NAMES[pred]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout(); plt.savefig(results_dir / "samples.png", dpi=150); plt.close()

    plot_history(results_dir)
    (results_dir / "test_metrics.json").write_text(
        json.dumps({"test_loss": test_loss, "test_accuracy": test_acc}, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--weights", default="models/best_cnn.pth")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())