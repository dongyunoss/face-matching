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
  { id: "horse",    ko: "말상",     emoji: "🐴", desc: "길고 갸름한 얼굴과 시원한 이목구비. 훤칠하고 시원시원한 인상이에요." },
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

// 재촬영 판정 기준 (구분이 불가능한 사진이면 결과 대신 재촬영 메시지를 띄움)
// 실제 카메라로 테스트하면서 값을 조정하세요.
export const QUALITY = {
  minFaceRatio: 0.2,    // 얼굴 너비 / 사진 짧은 변. 이보다 작으면 "더 가까이"
  maxYaw: 0.3,          // 좌우로 고개 돌린 정도(0 = 정면). 이보다 크면 "정면을 봐 주세요"
  minBrightness: 60,    // 얼굴 평균 밝기(0~255). 이보다 어두우면 "더 밝은 곳에서"
  maxBrightness: 220,   // 이보다 밝으면 "빛이 너무 강해요"
  minSharpness: 15,     // 선명도(라플라시안 분산). 이보다 낮으면 "흔들렸어요"
  minConfidence: 0.2,   // 모델의 최고 확률. 이보다 낮으면 "얼굴상을 구분하기 어려워요"
};

// ───────── 추천 · 매칭 규칙 (자세한 정의는 README 의 "추천·매칭 시스템" 참고) ─────────
// 만족도 항목. 각 사람이 항목별 중요도를 정하고, 그 가중치로 '내가 상대에게 얼마나 만족하는지'를 계산한다.
export const FACTORS = [
  { id: "face",     ko: "외모(동물상)" },
  { id: "mbti",     ko: "성격(MBTI)" },
  { id: "interest", ko: "관심사" },
  { id: "distance", ko: "거리" },
];
// 중요도 단계 -> 가중치. '매우 중요'가 '조금'의 6배.
export const IMPORTANCE_LEVELS = [
  { w: 0, ko: "상관없음" },
  { w: 1, ko: "조금" },
  { w: 3, ko: "중요" },
  { w: 6, ko: "매우 중요" },
];
export const DEFAULT_IMPORTANCE = { face: 3, mbti: 1, interest: 3, distance: 1 };

// 이상형 1·2·3순위 가중치
export const PREF_RANK_WEIGHTS = [0.5, 0.3, 0.2];

// 거리 점수: 같은 시·도 1.0, 같은 권역 0.7, 그 외 0.3
export const REGION_GROUPS = {
  수도권: ["서울", "경기", "인천"],
  강원: ["강원"],
  충청: ["대전", "세종", "충북", "충남"],
  호남: ["광주", "전북", "전남"],
  영남: ["부산", "대구", "울산", "경북", "경남"],
  제주: ["제주"],
};
export const DISTANCE_SCORE = { same: 1.0, group: 0.7, far: 0.3 };

// 취향 학습: 좋아요 n개를 받으면 이상형 벡터 = (1-β)·고른 이상형 + β·좋아요한 얼굴 평균,  β = n / (n + LEARN_K)
export const LEARN_K = 5;

// 한 번에 보여 주는 추천 수
export const DECK_SIZE = 5;

// 데모 후보의 가상 응답: 상대가 나에게 느끼는 만족도 sat(상대→나)가 이 값 이상이면 상대도 좋아요를 눌렀다고 본다.
// 시뮬레이션 기준 추천된 상대의 약 4명 중 1명이 이 조건을 넘는다.
export const DEMO_LIKE_BACK = 0.45;

export const MBTI_TYPES = [
  "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
  "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
];

export const HOBBIES = ["운동", "여행", "맛집", "카페", "영화", "음악", "게임", "독서", "사진", "요리", "반려동물", "등산", "술", "전시", "집콕"];

// 자기소개에서 관심사를 찾는 키워드 사전 (태그 이름은 HOBBIES 와 같음)
// training/make_demo_candidates.py 의 자기소개 문장도 이 키워드를 쓰도록 맞춰 두었다.
export const INTEREST_KEYWORDS = {
  운동: ["운동", "헬스", "러닝", "달리기", "필라테스", "요가", "클라이밍", "테니스", "축구", "농구", "수영", "배드민턴", "볼링"],
  여행: ["여행", "해외", "제주", "비행기", "캠핑", "드라이브", "바다"],
  맛집: ["맛집", "먹방", "미식", "디저트", "맛있는", "빵집", "웨이팅"],
  카페: ["카페", "커피", "라떼", "브런치"],
  영화: ["영화", "넷플", "드라마", "OTT", "시리즈", "애니"],
  음악: ["음악", "노래", "콘서트", "페스티벌", "공연", "기타", "피아노", "플레이리스트", "밴드"],
  게임: ["게임", "롤", "닌텐도", "스위치", "보드게임", "PC방", "스팀"],
  독서: ["독서", "책", "소설", "에세이", "서점"],
  사진: ["사진", "카메라", "필름", "인생샷"],
  요리: ["요리", "베이킹", "집밥", "홈쿡"],
  반려동물: ["반려동물", "반려견", "반려묘", "댕댕", "냥이", "집사", "멍멍"],
  등산: ["등산", "트레킹", "하이킹", "산행"],
  술: ["와인", "맥주", "소주", "칵테일", "이자카야", "한잔", "술"],
  전시: ["전시", "미술관", "박물관", "팝업", "갤러리"],
  집콕: ["집순이", "집돌이", "집콕", "집에서"],
};

// 전국 17개 시·도 (데모 후보도 이 목록에서 고르게 배정)
export const REGIONS = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"];

// ───────── Supabase (2단계: 회원가입 + DB) ─────────
// Supabase 대시보드 → Project Settings → API 에서 복사. 비워 두면 데모 모드로만 동작.
// anon(publishable) key 는 공개돼도 되는 키이며, 실제 보호는 supabase/schema.sql 의 RLS 정책이 담당한다.
export const SUPABASE_URL = "";
export const SUPABASE_ANON_KEY = "";
