"""MediaPipe Face Landmarker 기반 얼굴 정렬·크롭.

웹(web/js/facecrop.js)과 '완전히 같은 수식'을 쓴다. 학습 이미지와 서비스 입력이
같은 방식으로 잘려야 실제 사용 시 정확도가 유지된다.

수식 요약
1) 두 눈 중심(m_L, m_R)을 잇는 선의 기울기 angle 을 구한다.
2) 모든 랜드마크를 -angle 만큼 회전(기준점: 두 눈의 중점 m)해 bbox 를 구한다.
3) bbox 중심 c, 한 변 side = max(w, h) * CROP_MARGIN 인 정사각형을 잡는다.
4) 출력 픽셀 o(0~223) -> 원본 좌표: src = R(angle) * (k*o + c0 - m) + m
   (k = side / 224, c0 = c - side/2)
"""
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import CROP_MARGIN, IMG_SIZE, LANDMARKER_URL

LEFT_EYE = (33, 133)    # 한쪽 눈의 양 끝
RIGHT_EYE = (362, 263)  # 다른 쪽 눈의 양 끝


def crop_params(landmarks_px: np.ndarray, out_size: int = IMG_SIZE, margin: float = CROP_MARGIN):
    """landmarks_px: (N,2) 픽셀 좌표. 반환: (2x3 affine: 출력->원본) 및 보조 정보."""
    le = landmarks_px[list(LEFT_EYE)].mean(axis=0)
    re = landmarks_px[list(RIGHT_EYE)].mean(axis=0)
    angle = math.atan2(re[1] - le[1], re[0] - le[0])
    m = (le + re) / 2.0
    cos, sin = math.cos(angle), math.sin(angle)
    r_inv = np.array([[cos, sin], [-sin, cos]])          # R(-angle)
    rot = (landmarks_px - m) @ r_inv.T + m                # 기울기 제거한 좌표
    x0, y0 = rot.min(axis=0)
    x1, y1 = rot.max(axis=0)
    c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    side = max(x1 - x0, y1 - y0) * margin
    k = side / out_size
    r = np.array([[cos, -sin], [sin, cos]])               # R(angle)
    a = k * r
    t = r @ (c - side / 2.0 - m) + m
    affine = np.hstack([a, t[:, None]])                   # src = A @ o + t
    return affine, dict(angle=angle, center=c, side=side, mid=m)


def warp_face(img_rgb: np.ndarray, affine: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    return cv2.warpAffine(img_rgb, affine, (out_size, out_size),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class FaceCropper:
    """사용 예: FaceCropper('face_landmarker.task').crop(rgb_numpy) -> 224x224 RGB 또는 None"""

    def __init__(self, model_path: str = "face_landmarker.task"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            print(f"[facecrop] 모델 다운로드: {LANDMARKER_URL}")
            urllib.request.urlretrieve(LANDMARKER_URL, model_path)
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def landmarks(self, img_rgb: np.ndarray):
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(img_rgb))
        res = self.landmarker.detect(mp_img)
        if not res.face_landmarks:
            return None
        h, w = img_rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)

    def crop(self, img_rgb: np.ndarray):
        lm = self.landmarks(img_rgb)
        if lm is None:
            return None
        affine, _ = crop_params(lm)
        return warp_face(img_rgb, affine)
