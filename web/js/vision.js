// 얼굴 검출(MediaPipe Face Landmarker) + 얼굴상 분류(ONNX Runtime Web). 모두 브라우저 안에서 실행된다.
import { CLASSES, IMG_SIZE, LANDMARKER_MODEL_URL, MEAN, MEDIAPIPE_VERSION, MODEL_URL, QUALITY, STD } from "./config.js";
import { drawAlignedFace } from "./facecrop.js";

const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
let landmarker = null;
let session = null;
export let mockModel = false;   // 모델 파일이 없을 때 임시 결과로 UI 개발을 계속할 수 있게 함

export async function initVision(onStatus = () => {}) {
  onStatus("얼굴 검출기 불러오는 중…");
  const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/+esm`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: LANDMARKER_MODEL_URL, delegate: "CPU" },
    runningMode: "IMAGE",
    numFaces: 2,   // 2명 이상 찍혔는지 확인하기 위해 2까지 검출
  });

  onStatus("얼굴상 분류 모델 불러오는 중…");
  const head = await fetch(MODEL_URL, { method: "HEAD" }).catch(() => null);
  if (!head || !head.ok) {
    mockModel = true;
  } else {
    window.ort.env.wasm.numThreads = 1;
    session = await window.ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  }
  onStatus("");
  return { mockModel };
}

/**
 * source(canvas/img) 에서 얼굴을 찾아 정렬 크롭하고, 구분 가능한 사진인지 검사한다.
 * 반환: { face: canvas | null, issue: 재촬영 사유 문자열 | null }
 */
export function detectAndCrop(source, outCanvas) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  const res = landmarker.detect(source);
  const faces = res.faceLandmarks || [];
  if (faces.length === 0) return { face: null, issue: "얼굴이 보이지 않아요. 얼굴이 화면 가운데 오도록 해 주세요." };
  if (faces.length > 1) return { face: null, issue: "여러 사람이 찍혔어요. 한 사람만 나오도록 해 주세요." };

  const pts = faces[0].map(p => [p.x * w, p.y * h]);
  const xs = pts.map(p => p[0]);
  const faceW = Math.max(...xs) - Math.min(...xs);
  if (faceW < QUALITY.minFaceRatio * Math.min(w, h)) {
    return { face: null, issue: "얼굴이 너무 작아요. 카메라에 조금 더 가까이 와 주세요." };
  }
  if (Math.abs(yawRatio(pts)) > QUALITY.maxYaw) {
    return { face: null, issue: "얼굴이 옆을 향하고 있어요. 카메라 정면을 바라봐 주세요." };
  }

  const face = drawAlignedFace(source, pts, outCanvas);
  const { brightness, sharpness } = imageStats(face);
  if (brightness < QUALITY.minBrightness) return { face: null, issue: "너무 어두워요. 밝은 곳에서 찍어 주세요." };
  if (brightness > QUALITY.maxBrightness) return { face: null, issue: "빛이 너무 강해요. 역광이나 강한 조명을 피해 주세요." };
  if (sharpness < QUALITY.minSharpness) return { face: null, issue: "사진이 흐릿해요. 움직이지 말고 찍어 주세요." };
  return { face, issue: null };
}

/** 모델 결과가 특정 동물상으로 구분되지 않을 만큼 애매하면 재촬영 사유를 반환 */
export function confidenceIssue(probs) {
  if (mockModel) return null;   // 가짜 결과일 때는 검사하지 않음
  return Math.max(...probs) < QUALITY.minConfidence
    ? "얼굴상을 구분하기 어려운 사진이에요. 앞머리·안경·손 등으로 가리지 않은 얼굴로 찍어 주세요."
    : null;
}

// 코끝이 두 눈 사이에서 얼마나 치우쳤는지 (0 = 정면, ±0.5 이상 = 크게 돌린 얼굴)
function yawRatio(pts) {
  const le = [(pts[33][0] + pts[133][0]) / 2, (pts[33][1] + pts[133][1]) / 2];
  const re = [(pts[362][0] + pts[263][0]) / 2, (pts[362][1] + pts[263][1]) / 2];
  const ex = re[0] - le[0], ey = re[1] - le[1];
  const len = Math.hypot(ex, ey) || 1;
  const mid = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
  const nose = pts[1];
  return ((nose[0] - mid[0]) * ex + (nose[1] - mid[1]) * ey) / (len * len);
}

// 크롭된 얼굴의 가운데 영역 평균 밝기와 선명도(라플라시안 분산)
function imageStats(canvas) {
  const { data } = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const g = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  const a = Math.round(IMG_SIZE * 0.25), b = Math.round(IMG_SIZE * 0.75);
  let sum = 0, n = 0, lsum = 0, lsq = 0;
  for (let y = a; y < b; y++) {
    for (let x = a; x < b; x++) {
      const i = y * IMG_SIZE + x;
      sum += g[i]; n++;
      const lap = g[i - 1] + g[i + 1] + g[i - IMG_SIZE] + g[i + IMG_SIZE] - 4 * g[i];
      lsum += lap; lsq += lap * lap;
    }
  }
  const lm = lsum / n;
  return { brightness: sum / n, sharpness: lsq / n - lm * lm };
}

/** 224x224 얼굴 canvas -> 클래스별 확률 배열 (길이 = 클래스 수) */
export async function classify(faceCanvas) {
  const { data } = faceCanvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const n = IMG_SIZE * IMG_SIZE;
  const input = new Float32Array(3 * n);           // HWC(RGBA) -> CHW, ImageNet 정규화
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) {
      input[ch * n + i] = (data[i * 4 + ch] / 255 - MEAN[ch]) / STD[ch];
    }
  }
  if (mockModel) return mockProbs(input);
  const tensor = new window.ort.Tensor("float32", input, [1, 3, IMG_SIZE, IMG_SIZE]);
  const out = await session.run({ input: tensor });
  return Array.from(out.probs.data);
}

// 모델이 없을 때만 쓰는 가짜 확률 (같은 사진 -> 같은 결과)
function mockProbs(input) {
  let seed = 0;
  for (let i = 0; i < input.length; i += 997) seed = (seed * 31 + Math.round(input[i] * 100)) | 0;
  const rand = () => ((seed = (seed * 1103515245 + 12345) | 0) >>> 0) / 4294967296;
  const v = CLASSES.map(() => Math.pow(rand(), 3));
  const s = v.reduce((a, b) => a + b, 0);
  return v.map(x => x / s);
}
