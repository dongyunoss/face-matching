"""[2~3일차] 라벨러들의 CSV를 합쳐 soft label 생성 + 일치도(Fleiss' kappa) 계산.

실행 예:
    python training/merge_labels.py --data-dir data            # data/labels/*.csv 를 모두 읽음
    python training/merge_labels.py --data-dir data --pilot-only   # 파일럿 결과만 분석

출력: <data-dir>/soft_labels.csv  (image, p_dog ... p_horse, n_votes, hard)
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import CLASSES, CLASSES_KO, DEFAULT_DATA_DIR, NUM_CLASSES


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (이미지 수, 클래스 수). 각 행의 합(라벨러 수)이 같아야 함."""
    n = counts.sum(axis=1)
    assert np.all(n == n[0]), "모든 이미지의 라벨러 수가 같아야 함"
    n = n[0]
    if n < 2:
        return float("nan")
    p_j = counts.sum(axis=0) / counts.sum()
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def confusing_pairs(counts: np.ndarray, top: int = 6):
    """같은 이미지에서 함께 표가 나온 클래스 쌍 = 라벨러들이 헷갈린 쌍"""
    co = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for c in counts:
        for i in range(NUM_CLASSES):
            for j in range(i + 1, NUM_CLASSES):
                co[i, j] += c[i] * c[j]
    pairs = sorted(((co[i, j], i, j) for i in range(NUM_CLASSES) for j in range(i + 1, NUM_CLASSES)), reverse=True)
    return [(CLASSES_KO[i], CLASSES_KO[j], int(v)) for v, i, j in pairs[:top] if v > 0]


def report(name, counts):
    if len(counts) == 0:
        return
    ns = counts.sum(axis=1)
    print(f"\n[{name}] 이미지 {len(counts)}장")
    for n_r in sorted(set(ns.tolist())):
        sub = counts[ns == n_r]
        if n_r >= 2 and len(sub) > 1:
            print(f"  라벨러 {n_r}명 이미지 {len(sub)}장 -> Fleiss' kappa = {fleiss_kappa(sub):.3f}")
    full_agree = (counts.max(axis=1) == ns).mean()
    print(f"  전원 일치 비율: {full_agree:.1%}")
    print("  자주 헷갈린 쌍:", confusing_pairs(counts))
    hard = counts.argmax(axis=1)
    dist = np.bincount(hard, minlength=NUM_CLASSES)
    print("  최다 득표 클래스 분포:", {CLASSES_KO[i]: int(d) for i, d in enumerate(dist)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--labels-glob", default=None, help="기본: <data-dir>/labels/*.csv")
    ap.add_argument("--min-votes", type=int, default=2, help="유효 표가 이 수 미만인 이미지는 제외")
    ap.add_argument("--pilot-only", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    phase = {}
    if (data_dir / "assignments.csv").exists():
        with open(data_dir / "assignments.csv", encoding="utf-8") as f:
            phase = {r["image"]: r["phase"] for r in csv.DictReader(f)}

    votes = {}  # (image, labeler) -> label   (같은 사람이 여러 번 내보내면 마지막 값)
    for path in sorted(glob.glob(args.labels_glob or str(data_dir / "labels" / "*.csv"))):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                votes[(r["image"].strip(), r["labeler"].strip())] = r["label"].strip()
    if not votes:
        raise SystemExit("라벨 CSV가 없습니다. <data-dir>/labels/ 에 labels_이름.csv 를 모아 주세요.")

    per_img = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    skipped = defaultdict(int)
    for (img, _), lab in votes.items():
        if args.pilot_only and phase.get(img) != "pilot":
            continue
        if lab in CLASSES:
            per_img[img][CLASSES.index(lab)] += 1
        else:
            skipped[img] += 1  # 'none'(판단 불가)

    imgs = sorted(per_img)
    counts = np.stack([per_img[i] for i in imgs]) if imgs else np.zeros((0, NUM_CLASSES))
    pilot_mask = np.array([phase.get(i) == "pilot" for i in imgs], dtype=bool)
    report("파일럿", counts[pilot_mask])
    if not args.pilot_only:
        report("본 라벨링", counts[~pilot_mask])
        report("전체", counts)

    if args.pilot_only:
        return
    out = data_dir / "soft_labels.csv"
    kept = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + [f"p_{c}" for c in CLASSES] + ["n_votes", "hard"])
        for img, c in zip(imgs, counts):
            n = int(c.sum())
            if n < args.min_votes:
                continue
            p = c / n
            w.writerow([img] + [f"{x:.4f}" for x in p] + [n, CLASSES[int(c.argmax())]])
            kept += 1
    print(f"\nsoft label 저장: {out}  ({kept}장, 표 부족으로 제외 {len(imgs) - kept}장)")


if __name__ == "__main__":
    main()
