"""데모용 가상 후보 40명 생성 -> web/js/candidates.js
실존 인물 없음. 사진 없이 동물 아이콘으로만 표시한다.
    python training/make_demo_candidates.py
"""
import json
import random

import numpy as np

from common import CLASSES, NUM_CLASSES, PROJECT_ROOT

HOBBIES = ["운동", "여행", "맛집", "카페", "영화", "음악", "게임", "독서", "사진", "요리", "반려동물", "등산"]
ADJ = ["산책하는", "커피좋아", "주말캠핑", "야경보는", "책읽는", "노래하는", "요리하는", "여행가는", "운동하는", "영화보는"]
NOUN = ["별", "구름", "바다", "숲", "달", "노을", "봄비", "햇살", "새벽", "파도"]
PREF_W = [0.5, 0.3, 0.2]


def main(n=40, seed=7):
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
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
        out.append(dict(
            id=f"demo-{i + 1:02d}", nickname=f"{r.choice(ADJ)}{r.choice(NOUN)}", gender=gender,
            age=age, seek="F" if gender == "M" else "M", ageMin=max(19, age - r.randint(3, 6)),
            ageMax=age + r.randint(3, 6), hobbies=r.sample(HOBBIES, r.randint(2, 4)),
            face=[round(float(x), 4) for x in face], pref=pref, faceType=CLASSES[int(face.argmax())],
        ))
    path = PROJECT_ROOT / "web" / "js" / "candidates.js"
    path.write_text("// 자동 생성 파일 (training/make_demo_candidates.py). 가상의 데모 후보이며 실존 인물이 아님.\n"
                    f"export const DEMO_CANDIDATES = {json.dumps(out, ensure_ascii=False, indent=1)};\n", encoding="utf-8")
    print(f"저장: {path} ({n}명)")


if __name__ == "__main__":
    main()
