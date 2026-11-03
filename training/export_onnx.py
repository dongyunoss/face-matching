"""[5일차] 학습된 ResNet18 -> ONNX 변환 (브라우저 onnxruntime-web 용).

모델 끝에 softmax 를 붙여서 내보내므로 웹에서는 출력이 바로 확률이다.
실행 예:
    python training/export_onnx.py --ckpt results/finetune_f100/model.pt
출력: web/model/face_type.onnx, web/model/meta.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from common import (CLASSES, CLASSES_KO, CROP_MARGIN, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, NUM_CLASSES,
                    PROJECT_ROOT)


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "results" / "finetune_f100" / "model.pt"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "web" / "model"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = WithSoftmax(m).eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = out_dir / "face_type.onnx"
    export_kw = dict(input_names=["input"], output_names=["probs"], opset_version=17)
    try:  # PyTorch 2.5+ : 기존(TorchScript) 내보내기 방식을 명시 -> 파일 하나로 저장됨
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kw)
    except TypeError:  # 구버전 PyTorch
        torch.onnx.export(model, dummy, str(onnx_path), **export_kw)
    if (out_dir / "face_type.onnx.data").exists():
        print("경고: 가중치가 face_type.onnx.data 로 분리 저장됨 -> 이 파일도 함께 web/model 에 두세요.")

    # 검증: PyTorch 와 ONNX Runtime 출력 비교
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    ref = np.concatenate([model(torch.from_numpy(x[i:i + 1])).detach().numpy() for i in range(2)])
    got = np.concatenate([sess.run(["probs"], {"input": x[i:i + 1]})[0] for i in range(2)])
    print(f"PyTorch vs ONNX 최대 오차: {np.abs(ref - got).max():.2e}")

    meta = dict(classes=CLASSES, classes_ko=CLASSES_KO, img_size=IMG_SIZE, crop_margin=CROP_MARGIN,
                mean=IMAGENET_MEAN, std=IMAGENET_STD)
    json.dump(meta, open(out_dir / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
