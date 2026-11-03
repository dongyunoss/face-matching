"""[4일차] 얼굴상 분류 학습 + 실험 (Colab GPU 권장).

모델
  mlp      : 강의(DL10)의 다층 퍼셉트론 baseline. 64x64 이미지를 펼쳐서 입력
  linear   : ImageNet 사전학습 ResNet18, 마지막 fc 층만 학습 (feature extraction)
  finetune : ImageNet 사전학습 ResNet18 전체 미세조정 (backbone lr 은 fc 의 1/10)

손실: soft label 교차 엔트로피  L = -Σ_k y_k log softmax(z)_k   (DL07 softmax 회귀의 일반화)

실행 예
  python training/train.py --data-dir data --exp all                  # 전체 실험 9개
  python training/train.py --data-dir data --model finetune --frac 1.0  # 하나만

출력: results/<run>/{history.json, metrics.json, curves.png, confusion.png, model.pt}
      results/summary.csv, results/datasize.png
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from common import CLASSES, DEFAULT_DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, NUM_CLASSES, PROJECT_ROOT

FRACS = [0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- 데이터
class FaceDataset(Dataset):
    def __init__(self, rows, img_dir, tf):
        self.rows, self.img_dir, self.tf = rows, Path(img_dir), tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tf(Image.open(self.img_dir / r["image"]).convert("RGB"))
        y = torch.tensor([float(r[f"p_{c}"]) for c in CLASSES], dtype=torch.float32)
        return x, y


def make_transforms(size):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return train_tf, eval_tf


def subsample(rows, frac, seed):
    """클래스 비율을 유지하며 학습 데이터 일부만 사용 (데이터 규모 실험)"""
    if frac >= 1.0:
        return rows
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["hard"], []).append(r)
    out = []
    for rs in by.values():
        rs = rs[:]
        rng.shuffle(rs)
        out += rs[: max(1, round(len(rs) * frac))]
    return out


# ---------------------------------------------------------------- 모델
def build_model(kind):
    if kind == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES))
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    if kind == "linear":
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
    return m


def make_optimizer(model, kind, lr):
    if kind == "finetune":
        fc = list(model.fc.parameters())
        fc_ids = {id(p) for p in fc}
        backbone = [p for p in model.parameters() if id(p) not in fc_ids]
        groups = [{"params": backbone, "lr": lr * 0.1}, {"params": fc, "lr": lr}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


# ---------------------------------------------------------------- 학습/평가
def soft_ce(logits, y):
    return F.cross_entropy(logits, y)  # PyTorch 는 확률 분포 형태의 target 을 지원


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += soft_ce(logits, y).item() * len(x)
        n += len(x)
        P.append(F.softmax(logits, dim=1).cpu())
        Y.append(y.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), loss_sum / max(n, 1)


def compute_metrics(P, Y, loss):
    pred, true = P.argmax(1), Y.argmax(1)
    labels = list(range(NUM_CLASSES))
    p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(true, pred, labels=labels, average=None, zero_division=0)
    eps = 1e-8
    kl = float(np.mean(np.sum(np.where(Y > 0, Y * (np.log(Y + eps) - np.log(P + eps)), 0), axis=1)))
    return {
        "loss": loss,
        "accuracy": float((pred == true).mean()),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1),
        # 예측한 클래스가 라벨러 중 1명 이상에게 표를 받은 비율 (주관적 과제용 보조 지표)
        "vote_hit_rate": float(np.mean(Y[np.arange(len(Y)), pred] > 0)),
        "kl_divergence": kl,
        "per_class": {c: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
    }


def plot_run(out_dir, hist, cm, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train"); ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_dir / "curves.png", dpi=120); plt.close(fig)

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (majority vote)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix - {title}"); fig.tight_layout(); fig.savefig(out_dir / "confusion.png", dpi=120)
    plt.close(fig)


def run(kind, frac, rows, args, device):
    name = f"{kind}_f{int(frac * 100):03d}"
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    size = 64 if kind == "mlp" else IMG_SIZE
    train_tf, eval_tf = make_transforms(size)
    img_dir = Path(args.data_dir) / "images"
    train_rows = subsample([r for r in rows if r["split"] == "train"], frac, args.seed)
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(FaceDataset(train_rows, img_dir, train_tf), shuffle=True, drop_last=len(train_rows) > args.batch_size, **kw)
    val_dl = DataLoader(FaceDataset(val_rows, img_dir, eval_tf), **kw)
    test_dl = DataLoader(FaceDataset(test_rows, img_dir, eval_tf), **kw)

    model = build_model(kind).to(device)
    epochs = args.epochs or {"mlp": 30, "linear": 15, "finetune": 15}[kind]
    opt = make_optimizer(model, kind, args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best, bad = float("inf"), 0
    print(f"\n=== {name}: train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)}, {epochs} epochs ===")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if kind == "linear":
            model.eval()  # 고정된 backbone 의 BatchNorm 통계가 바뀌지 않도록
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = soft_ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x); tn += len(x)
            tc += (logits.argmax(1) == y.argmax(1)).sum().item()
        sched.step()
        P, Y, vl = predict(model, val_dl, device)
        va = float((P.argmax(1) == Y.argmax(1)).mean())
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"  ep {ep:2d}  train loss {tl / tn:.4f} acc {tc / tn:.3f} | val loss {vl:.4f} acc {va:.3f}")
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  조기 종료 (val loss {args.patience} epoch 연속 미개선)")
                break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    P, Y, tl = predict(model, test_dl, device)
    m = compute_metrics(P, Y, tl)
    m.update(run=name, model=kind, frac=frac, n_train=len(train_rows), epochs_run=len(hist["train_loss"]),
             seconds=round(time.time() - t0, 1))
    json.dump(hist, open(out_dir / "history.json", "w"), indent=1)
    json.dump(m, open(out_dir / "metrics.json", "w"), indent=1, ensure_ascii=False)
    plot_run(out_dir, hist, m["confusion_matrix"], name)
    print(f"  TEST acc {m['accuracy']:.3f}  F1 {m['f1_macro']:.3f}  vote-hit {m['vote_hit_rate']:.3f}  KL {m['kl_divergence']:.3f}")
    return m


def write_summary(results, out_dir):
    out_dir = Path(out_dir)
    keys = ["run", "model", "frac", "n_train", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "vote_hit_rate", "kl_divergence", "loss", "epochs_run", "seconds"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for metric, a in zip(["accuracy", "f1_macro"], ax):
        for kind, mk in [("linear", "o-"), ("finetune", "s-"), ("mlp", "^")]:
            rs = sorted([r for r in results if r["model"] == kind], key=lambda r: r["n_train"])
            if rs:
                a.plot([r["n_train"] for r in rs], [r[metric] for r in rs], mk, label=kind)
        a.set_xlabel("# training images"); a.set_title(f"Test {metric} vs data size"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "datasize.png", dpi=120); plt.close(fig)
    print(f"\n요약 저장: {out_dir / 'summary.csv'}, {out_dir / 'datasize.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--exp", choices=["all", "single"], default="single")
    ap.add_argument("--model", choices=["mlp", "linear", "finetune"], default="finetune")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=0, help="0이면 모델별 기본값")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    with open(Path(args.data_dir) / "dataset.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.exp == "all":
        plan = [("mlp", 1.0)] + [("linear", fr) for fr in FRACS] + [("finetune", fr) for fr in FRACS]
    else:
        plan = [(args.model, args.frac)]
    results = [run(k, fr, rows, args, device) for k, fr in plan]

    summary_path = Path(args.out_dir) / "summary.csv"
    if args.exp == "single" and summary_path.exists():  # 기존 결과에 이어 붙이기
        old = [{**r, "frac": float(r["frac"]), "n_train": int(r["n_train"]),
                **{k: float(r[k]) for k in ["accuracy", "f1_macro"]}}
               for r in csv.DictReader(open(summary_path)) if r["run"] not in {x["run"] for x in results}]
        results = old + results
    write_summary(results, args.out_dir)


if __name__ == "__main__":
    main()
