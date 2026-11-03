"""[1일차] FairFace에서 East Asian, 20~39세, 남녀 반반을 뽑아 MediaPipe로 크롭해 저장.

Colab 실행 예:
    python training/prepare_fairface.py --data-dir /content/drive/MyDrive/faceproj/data --n 1500

출력
    <data-dir>/images/ff_00001.jpg ...   (224x224 정렬·크롭 이미지)
    <data-dir>/meta.csv                  (image, gender, age, fairface_index)
"""
import argparse
import csv
import io

import numpy as np
from PIL import Image

from common import DEFAULT_DATA_DIR
from facecrop import FaceCropper

TARGET_RACE = "East Asian"
TARGET_AGES = {"20-29", "30-39"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--n", type=int, default=1500, help="최종 이미지 수 (남녀 반반)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--landmarker", default="face_landmarker.task")
    args = ap.parse_args()

    from datasets import load_dataset  # pip install datasets

    from pathlib import Path
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1.25 = 얼굴 주변 여백이 넓은 버전 (MediaPipe가 얼굴 전체를 잡기 좋음)
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train", streaming=True)
    feats = ds.features
    if feats is not None:
        race_name, age_name, gender_name = (feats["race"].int2str, feats["age"].int2str,
                                            feats["gender"].int2str)
    else:  # 메타데이터가 없을 때 대비 (데이터셋 카드 기준 순서)
        races = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
        ages = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
        race_name, age_name = races.__getitem__, ages.__getitem__
        gender_name = ["Male", "Female"].__getitem__
    ds = ds.shuffle(seed=args.seed, buffer_size=5000)

    cropper = FaceCropper(args.landmarker)
    quota = {"Male": args.n // 2, "Female": args.n - args.n // 2}
    got = {"Male": 0, "Female": 0}
    rows, seen, fail = [], 0, 0

    for idx, ex in enumerate(ds):
        seen += 1
        if race_name(ex["race"]) != TARGET_RACE or age_name(ex["age"]) not in TARGET_AGES:
            continue
        g = gender_name(ex["gender"])
        if got[g] >= quota[g]:
            if all(got[k] >= quota[k] for k in got):
                break
            continue
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]))
        face = cropper.crop(np.array(img.convert("RGB")))
        if face is None:
            fail += 1
            continue
        got[g] += 1
        name = f"ff_{len(rows) + 1:05d}.jpg"
        Image.fromarray(face).save(img_dir / name, quality=95)
        rows.append({"image": name, "gender": g, "age": age_name(ex["age"]), "fairface_index": idx})
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{args.n} 저장 (스캔 {seen}장, 검출 실패 {fail}장)")

    with open(data_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "gender", "age", "fairface_index"])
        w.writeheader()
        w.writerows(rows)
    print(f"완료: {len(rows)}장 저장 -> {img_dir}  (남 {got['Male']}, 여 {got['Female']}, 검출 실패 {fail})")


if __name__ == "__main__":
    main()
