"""[1일차] FairFace에서 East Asian, 20~39세, 남녀 반반을 뽑아 MediaPipe로 크롭해 저장.

Colab 실행 예:
    python training/prepare_fairface.py --data-dir /content/drive/MyDrive/faceproj/data --n 1500

동작
  1) FairFace(1.25 버전) train 분할 전체를 내려받는다 (약 수 GB, Colab 에서 수 분).
  2) FairFace 가 제공하는 race / age / gender 라벨로 후보를 고른다 (이미지는 아직 열지 않음).
  3) 성별별 후보 순서를 seed 로 무작위로 섞고, 앞에서부터 얼굴 검출에 성공한 사진을 채택한다.

출력
    <data-dir>/images/ff_00001.jpg ...   (224x224 정렬·크롭 이미지)
    <data-dir>/meta.csv                  (image, fairface_index, race, gender, age)
        fairface_index = FairFace train 분할의 원본 행 번호 -> ds[fairface_index] 로 원본 확인 가능
"""
import argparse
import csv
from pathlib import Path

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

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("FairFace 내려받는 중 (처음 한 번만, 수 분 소요)…")
    ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")
    age_name = ds.features["age"].int2str

    # 라벨 열만 읽어서 후보 선별 (이미지 디코딩 없이 빠름)
    races = np.array(ds["race"])
    ages = np.array(ds["age"])
    genders = np.array(ds["gender"])
    race_id = ds.features["race"].str2int(TARGET_RACE)
    age_ids = {ds.features["age"].str2int(a) for a in TARGET_AGES}
    cand = np.where((races == race_id) & np.isin(ages, list(age_ids)))[0]
    print(f"train {len(ds)}장 중 조건(East Asian, 20~39세) 해당: {len(cand)}장")

    rng = np.random.default_rng(args.seed)
    cropper = FaceCropper(args.landmarker)
    quota = {"Male": args.n // 2, "Female": args.n - args.n // 2}
    picked = []  # (fairface_index, gender, age, face)
    fail = 0
    for g, q in quota.items():
        gid = ds.features["gender"].str2int(g)
        order = rng.permutation(cand[genders[cand] == gid])
        print(f"  {g}: 후보 {len(order)}장 중 {q}장 선택")
        got = 0
        for idx in order:
            if got >= q:
                break
            ex = ds[int(idx)]
            face = cropper.crop(np.array(ex["image"].convert("RGB")))
            if face is None:
                fail += 1
                continue
            picked.append((int(idx), g, age_name(ex["age"]), face))
            got += 1
        if got < q:
            print(f"  경고: {g} 후보가 부족해 {got}장만 선택됨")

    picked.sort(key=lambda r: r[0])  # 원본 순서대로 파일 번호 부여
    rows = []
    for k, (idx, g, age, face) in enumerate(picked, start=1):
        name = f"ff_{k:05d}.jpg"
        Image.fromarray(face).save(img_dir / name, quality=95)
        rows.append({"image": name, "fairface_index": idx, "race": TARGET_RACE, "gender": g, "age": age})

    with open(data_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "fairface_index", "race", "gender", "age"])
        w.writeheader()
        w.writerows(rows)
    n_m = sum(r["gender"] == "Male" for r in rows)
    print(f"완료: {len(rows)}장 저장 -> {img_dir}  (남 {n_m}, 여 {len(rows) - n_m}, 얼굴 검출 실패로 건너뜀 {fail})")


if __name__ == "__main__":
    main()
