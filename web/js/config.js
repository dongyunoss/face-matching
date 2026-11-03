// ───────── 프로젝트 설정 ─────────
// 클래스 순서 = 모델 출력 순서. training/common.py 의 CLASSES 와 반드시 같아야 한다.
export const CLASSES = [
  { id: "dog",      ko: "강아지상", emoji: "🐶", desc: "순하고 다정한 인상. 처음 만나도 편안함을 주는 타입이에요." },
  { id: "cat",      ko: "고양이상", emoji: "🐱", desc: "도도하고 세련된 인상. 올라간 눈매와 날렵한 선이 매력이에요." },
  { id: "rabbit",   ko: "토끼상",   emoji: "🐰", desc: "동그란 눈과 생기 있는 표정. 귀엽고 밝은 인상이에요." },
  { id: "fox",      ko: "여우상",   emoji: "🦊", desc: "가늘고 긴 눈매와 갸름한 얼굴. 여유 있고 매혹적인 인상이에요." },
  { id: "bear",     ko: "곰상",     emoji: "🐻", desc: "둥글고 푸근한 인상. 듬직하고 따뜻한 느낌을 줘요." },
  { id: "deer",     ko: "사슴상",   emoji: "🦌", desc: "크고 맑은 눈. 청순하고 차분한 인상이에요." },
  { id: "dino",     ko: "공룡상",   emoji: "🦖", desc: "선이 굵고 큰 이목구비. 개성 있고 시원한 인상이에요." },
  { id: "wolf",     ko: "늑대상",   emoji: "🐺", desc: "날카로운 눈매와 진한 눈썹. 강하고 카리스마 있는 인상이에요." },
  { id: "squirrel", ko: "다람쥐상", emoji: "🐿️", desc: "작고 동그란 얼굴과 통통한 볼. 앙증맞고 귀여운 인상이에요." },
];

// 얼굴 크롭·정규화 (training/common.py 와 동일)
export const IMG_SIZE = 224;
export const CROP_MARGIN = 1.3;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];

export const MODEL_URL = "./model/face_type.onnx";
export const MEDIAPIPE_VERSION = "0.10.14";
export const LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// 매칭 점수 가중치 (합 = 1)
export const MATCH_WEIGHTS = { mutual: 0.8, hobby: 0.2 };
// 이상형 1·2·3순위 가중치
export const PREF_RANK_WEIGHTS = [0.5, 0.3, 0.2];

export const HOBBIES = ["운동", "여행", "맛집", "카페", "영화", "음악", "게임", "독서", "사진", "요리", "반려동물", "등산"];

// ───────── Supabase (2단계: 회원가입 + DB) ─────────
// Supabase 대시보드 → Project Settings → API 에서 복사. 비워 두면 데모 모드로만 동작.
// anon(publishable) key 는 공개돼도 되는 키이며, 실제 보호는 supabase/schema.sql 의 RLS 정책이 담당한다.
export const SUPABASE_URL = "";
export const SUPABASE_ANON_KEY = "";
