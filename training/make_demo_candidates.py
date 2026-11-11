"""데모용 가상 후보 40명 생성 -> web/js/candidates.js
실존 인물 없음. 사진 없이 동물 아이콘으로만 표시한다.
    python training/make_demo_candidates.py
"""
import json
import random

import numpy as np

from common import CLASSES, NUM_CLASSES, PROJECT_ROOT

HOBBIES = ["운동", "여행", "맛집", "카페", "영화", "음악", "게임", "독서", "사진", "요리", "반려동물", "등산"]
MBTI = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"]
REGIONS = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]  # 전국 17개 시·도
# 자기소개 문장 조각. web/js/config.js 의 INTEREST_KEYWORDS 에 있는 단어를 쓴다.
HOBBY_LINE = {
    "운동": "퇴근하고 헬스 가는 게 루틴이에요.", "여행": "한 달에 한 번은 여행 가려고 해요.",
    "맛집": "웨이팅 있는 맛집도 같이 기다려 줄 사람!", "카페": "주말엔 동네 카페 탐방해요.",
    "영화": "넷플 보면서 쉬는 게 최고의 힐링.", "음악": "콘서트랑 페스티벌 좋아해요.",
    "게임": "닌텐도 같이 할 사람 구해요.", "독서": "요즘 에세이 읽는 재미에 빠졌어요.",
    "사진": "필름 카메라로 일상 찍는 걸 좋아해요.", "요리": "집밥 해 먹는 걸 좋아해요.",
    "반려동물": "댕댕이랑 산책하는 게 하루의 낙이에요.", "등산": "주말 아침 등산 다녀요.",
}
EXTRA_LINE = ["저녁엔 와인 한잔 좋아해요.", "전시나 팝업 구경 다니는 거 좋아해요.", "쉬는 날엔 집순이 모드예요.",
              "드라이브 하면서 바다 보러 가는 거 좋아해요.", "", "", ""]
STYLE_LINE = ["연락 자주 하는 편이에요 :)", "편하게 대화부터 해요!", "웃음 코드 맞는 사람 좋아요.",
              "천천히 알아가고 싶어요.", "솔직하고 다정한 사람이 좋아요.", "같이 맛있는 거 먹으러 가요!"]
ADJ = ["산책하는", "커피좋아", "주말캠핑", "야경보는", "책읽는", "노래하는", "요리하는", "여행가는", "운동하는", "영화보는"]
NOUN = ["별", "구름", "바다", "숲", "달", "노을", "봄비", "햇살", "새벽", "파도"]
PREF_W = [0.5, 0.3, 0.2]


def main(n=40, seed=7):
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    r2 = random.Random(seed + 1)   # MBTI·자기소개용 (기존 얼굴/이상형 값이 바뀌지 않도록 별도 시드)
    r3 = random.Random(seed + 2)   # 항목별 중요도용
    out = []
    for i in range(n):
        gender = "M" if i % 2 == 0 else "F"
        face = rng.dirichlet(np.full(NUM_CLASSES, 0.35))
        ranks = r.sample(range(NUM_CLASSES), r.randint(1, 3))
        w = PREF_W[: len(ranks)]
        pref = [0.0] * NUM_CLASSES
        for k, c in enumerate(ranks):
            pref[c] = round(w[k] / sum(w), 4)
        age = r.randint(21, 34)
        hobbies = r.sample(HOBBIES, r.randint(2, 4))
        mbti = r2.choice(MBTI)
        pref_mbti = r2.sample(MBTI, r2.choice([0, 0, 1, 2]))
        bio = " ".join(x for x in [HOBBY_LINE[hobbies[0]], r2.choice(EXTRA_LINE), r2.choice(STYLE_LINE)] if x)
        out.append(dict(
            id=f"demo-{i + 1:02d}", nickname=f"{r.choice(ADJ)}{r.choice(NOUN)}", gender=gender,
            age=age, seek="F" if gender == "M" else "M", ageMin=max(19, age - r.randint(3, 6)),
            ageMax=age + r.randint(3, 6), hobbies=hobbies,
            face=[round(float(x), 4) for x in face], pref=pref, faceType=CLASSES[int(face.argmax())],
            mbti=mbti, prefMbti=pref_mbti, bio=bio, region=REGIONS[i % len(REGIONS)],   # 40명을 17개 시·도에 고르게 배정
            # 항목별 중요도 가중치 (0 상관없음 / 1 조금 / 3 중요 / 6 매우 중요) — web/js/config.js 의 IMPORTANCE_LEVELS
            importance=dict(face=r3.choice([1, 3, 3, 6]), mbti=r3.choice([0, 1, 1, 3]),
                            interest=r3.choice([1, 3, 3, 6]), distance=r3.choice([0, 1, 1, 3])),
        ))
    path = PROJECT_ROOT / "web" / "js" / "candidates.js"
    path.write_text("// 자동 생성 파일 (training/make_demo_candidates.py). 가상의 데모 후보이며 실존 인물이 아님.\n"
                    f"export const DEMO_CANDIDATES = {json.dumps(out, ensure_ascii=False, indent=1)};\n", encoding="utf-8")
    print(f"저장: {path} ({n}명)")


if __name__ == "__main__":
    main()
