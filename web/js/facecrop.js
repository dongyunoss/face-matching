// MediaPipe 랜드마크 기반 얼굴 정렬·크롭.
// training/facecrop.py 와 '같은 수식'이다. 학습 이미지와 서비스 입력을 똑같이 잘라야 정확도가 유지된다.
import { CROP_MARGIN, IMG_SIZE } from "./config.js";

const LEFT_EYE = [33, 133];
const RIGHT_EYE = [362, 263];

const mean2 = (pts, ids) => [(pts[ids[0]][0] + pts[ids[1]][0]) / 2, (pts[ids[0]][1] + pts[ids[1]][1]) / 2];

/** pts: [[x,y], ...] 픽셀 좌표 (478개). 반환: 출력->원본 affine 파라미터 */
export function cropParams(pts, outSize = IMG_SIZE, margin = CROP_MARGIN) {
  const le = mean2(pts, LEFT_EYE), re = mean2(pts, RIGHT_EYE);
  const angle = Math.atan2(re[1] - le[1], re[0] - le[0]);
  const m = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
  const cos = Math.cos(angle), sin = Math.sin(angle);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {            // R(-angle) 로 기울기 제거
    const dx = x - m[0], dy = y - m[1];
    const rx = cos * dx + sin * dy + m[0];
    const ry = -sin * dx + cos * dy + m[1];
    x0 = Math.min(x0, rx); x1 = Math.max(x1, rx); y0 = Math.min(y0, ry); y1 = Math.max(y1, ry);
  }
  const c = [(x0 + x1) / 2, (y0 + y1) / 2];
  const side = Math.max(x1 - x0, y1 - y0) * margin;
  const k = side / outSize;
  const c0 = [c[0] - side / 2 - m[0], c[1] - side / 2 - m[1]];
  const t = [cos * c0[0] - sin * c0[1] + m[0], sin * c0[0] + cos * c0[1] + m[1]];
  return { cos, sin, k, t, angle };      // src = k*R(angle)*o + t
}

/** 원본(canvas/img/video)에서 정렬된 얼굴을 outCanvas(224x224)에 그린다 */
export function drawAlignedFace(source, pts, outCanvas) {
  const { cos, sin, k, t } = cropParams(pts);
  outCanvas.width = outCanvas.height = IMG_SIZE;
  const ctx = outCanvas.getContext("2d", { willReadFrequently: true });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  // 역변환: o = (1/k) R^T (src - t)   (+ 픽셀 중심 0.5 보정으로 OpenCV warpAffine 과 일치)
  const a = cos / k, b = -sin / k, c = sin / k, d = cos / k;
  let e = -(cos * t[0] + sin * t[1]) / k;
  let f = (sin * t[0] - cos * t[1]) / k;
  e += 0.5 - (a + c) * 0.5;
  f += 0.5 - (b + d) * 0.5;
  ctx.setTransform(a, b, c, d, e, f);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return outCanvas;
}
