// 얼굴 검출(MediaPipe Face Landmarker) + 얼굴상 분류(ONNX Runtime Web). 모두 브라우저 안에서 실행된다.
import { CLASSES, IMG_SIZE, LANDMARKER_MODEL_URL, MEAN, MEDIAPIPE_VERSION, MODEL_URL, STD } from "./config.js";
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
    numFaces: 1,
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

/** source(canvas/img) 에서 얼굴을 찾아 정렬 크롭. 얼굴이 없으면 null */
export function detectAndCrop(source, outCanvas) {
  const w = source.naturalWidth || source.videoWidth || source.width;
  const h = source.naturalHeight || source.videoHeight || source.height;
  const res = landmarker.detect(source);
  if (!res.faceLandmarks || res.faceLandmarks.length === 0) return null;
  const pts = res.faceLandmarks[0].map(p => [p.x * w, p.y * h]);
  return drawAlignedFace(source, pts, outCanvas);
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
