"""[2일차] 라벨러별 라벨링 분량 배정.

- 파일럿: 앞쪽 --pilot 장은 라벨러 전원이 라벨링 (일치도 측정용)
- 본 라벨링: 나머지 이미지는 각각 --per-image 명이 라벨링 (순환 배정 -> 1인당 분량이 균등)

실행 예:
    python training/make_assignments.py --data-dir data --labelers 주환 정원 정유 동윤 채하

출력: <data-dir>/labeling/<라벨러>/pilot/*.jpg, <data-dir>/labeling/<라벨러>/main/*.jpg
      -> 폴더째 압축해서 각자에게 전달 (구글 드라이브 등)
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from common import DEFAULT_DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--labelers", nargs="+", required=True)
    ap.add_argument("--pilot", type=int, default=50)
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "meta.csv", encoding="utf-8") as f:
        images = [r["image"] for r in csv.DictReader(f)]
    random.Random(args.seed).shuffle(images)
    pilot, main_imgs = images[: args.pilot], images[args.pilot:]
    L = len(args.labelers)
    assert args.per_image <= L, "--per-image 는 라벨러 수 이하여야 함"

    plan = []  # (image, labeler, phase)
    for img in pilot:
        plan += [(img, name, "pilot") for name in args.labelers]
    for i, img in enumerate(main_imgs):
        plan += [(img, args.labelers[(i + j) % L], "main") for j in range(args.per_image)]

    out = data_dir / "labeling"
    for img, name, phase in plan:
        d = out / name / phase
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / "images" / img, d / img)

    with open(data_dir / "assignments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "labeler", "phase"])
        w.writerows(plan)

    for name in args.labelers:
        n_p = sum(1 for _, n, p in plan if n == name and p == "pilot")
        n_m = sum(1 for _, n, p in plan if n == name and p == "main")
        print(f"{name}: 파일럿 {n_p}장, 본 라벨링 {n_m}장")
    print(f"폴더 생성 완료 -> {out}")


if __name__ == "__main__":
    main()
