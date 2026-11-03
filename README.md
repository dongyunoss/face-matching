# 🐾 동물상 소개팅 — 딥러닝 얼굴상 분류 기반 매칭 웹

얼굴 사진을 9가지 동물상(강아지·고양이·토끼·여우·곰·사슴·공룡·늑대·다람쥐)으로 분류하고,
서로의 이상형을 함께 반영한 **상호 매칭 점수**로 소개팅 상대 Top 5를 추천합니다.

- **모델**: ImageNet으로 사전학습된 ResNet18을 미세조정(fine-tuning)합니다. 라벨러 투표 비율(soft label)로 학습해요.
- **추론**: 브라우저에서 실행합니다(MediaPipe 얼굴 검출 + ONNX Runtime Web). 사진은 서버로 전송되지 않아요.
- **화면 표시**: 동물상 하나만 보여 줍니다. 매칭 계산에는 9개 클래스의 확률 벡터 전체를 씁니다.
- **매칭**
  - 1단계: 데모용 가상 후보 40명 중에서 추천해요.
  - 2단계: Supabase로 회원가입하고, 실제 회원 중에서 추천해요.

---

## 폴더 구조

```
face-matching/
├─ training/                 # Python (Colab에서 실행)
│  ├─ common.py              # 클래스 목록, 크롭/정규화 상수 (web/js/config.js와 같아야 함)
│  ├─ facecrop.py            # MediaPipe 얼굴 정렬·크롭 (web/js/facecrop.js와 같은 수식)
│  ├─ prepare_fairface.py    # 1일차: FairFace 추출 + 크롭
│  ├─ make_assignments.py    # 2일차: 라벨러별 분량 배정
│  ├─ merge_labels.py        # 2~3일차: soft label 생성 + Fleiss' kappa
│  ├─ split_dataset.py       # 3일차: 중복 검사 + train/val/test 분리
│  ├─ train.py               # 4일차: 학습 + 실험 (MLP / linear probe / fine-tune × 데이터 규모)
│  ├─ export_onnx.py         # 5일차: ONNX 변환
│  └─ make_demo_candidates.py# 데모 후보 생성
├─ notebooks/colab_pipeline.ipynb   # 위 스크립트를 순서대로 실행하는 Colab 노트북
├─ labeler/index.html        # 라벨링 도구 (설치 없이 브라우저로 열기)
├─ web/                      # 배포할 웹사이트 (빌드 과정 없음)
│  ├─ index.html, css/, js/
│  └─ model/                 # 여기에 face_type.onnx, meta.json을 넣음
└─ supabase/schema.sql       # 2단계 DB 테이블 + 보안 정책
```

## 사용하는 도구

| 용도 | 도구 |
|---|---|
| 코드 편집 | VS Code (확장 프로그램 **Live Server** 추천) |
| 코드 공유 | GitHub (저장소 1개를 팀 전체가 같이 씀) |
| 데이터 준비 / 학습 | **Google Colab** (무료 T4 GPU) + 구글 드라이브 |
| 라벨링 | `labeler/index.html` (자체 도구: 키보드 1~9로 빠르게 라벨링, 설치 불필요) |
| 배포 | **Render Static Site** (잠들지 않음, HTTPS 자동 → 카메라 사용 가능) |
| 회원 / DB | **Supabase** (무료 Postgres + 로그인) |

처음에는 Label Studio와 Vite를 추천했지만, 일주일 일정에 맞춰 둘 다 더 단순한 방식으로 바꿨어요.

- **라벨링 도구**: Label Studio는 각자 설치하고 업로드해야 해요. 자체 도구는 HTML 파일을 더블클릭하면 바로 열려요.
- **웹 프레임워크**: Vite 같은 빌드 도구 없이 CDN으로 라이브러리를 불러와요. 그래서 `npm install`도 빌드도 필요 없어요.

---

## 일정별 진행 방법

### 0일차: 저장소 만들기 (팀장)
1. 이 폴더를 VS Code로 열어요.
2. GitHub에 `face-matching` 저장소를 만들고 push해요.
   ```bash
   git init && git add . && git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/<아이디>/face-matching.git
   git push -u origin main
   ```
3. 팀원을 Collaborator로 초대해요(Settings → Collaborators).
4. `notebooks/colab_pipeline.ipynb`의 `REPO_URL`을 저장소 주소로 바꿔요.
   - 저장소가 비공개라면 Colab에서 clone할 때 토큰이 필요해요. 그래서 **공개 저장소를 권장**해요.
   - 데이터와 결과는 `.gitignore`로 제외돼 있어서 올라가지 않아요.

### 1일차: 데이터 준비 (Colab)
1. GitHub의 노트북 파일을 Colab에서 열어요(colab.research.google.com → GitHub 탭).
2. 런타임 유형을 **T4 GPU**로 바꿔요.
3. 셀을 위에서부터 실행해요.
   - `prepare_fairface.py`가 FairFace에서 East Asian, 20~39세, 남녀 750명씩 총 1,500장을 뽑아요.
   - 뽑은 사진은 MediaPipe로 **눈높이를 맞춰 정렬하고 224×224 크기로 잘라** 구글 드라이브에 저장해요.
4. 크롭 결과 미리보기 셀로 얼굴이 제대로 잘렸는지 확인해요.
5. 같은 날 팀 회의에서 `labeler/index.html` 오른쪽의 **판단 기준표**를 함께 다듬어요(파일 안의 `CLASSES` 배열).

### 2일차: 라벨링
1. Colab에서 `make_assignments.py`를 실행해요(팀원 이름은 노트북에서 수정).
   - 결과는 `data/labeling/<이름>/pilot`과 `.../main` 폴더로 나뉘어요.
   - 파일럿 50장은 5명 모두가 라벨링해요.
   - 나머지 1,450장은 장당 3명이 라벨링해요. 1인당 약 870장이에요.
2. 각자 드라이브에서 자기 폴더를 내려받아요.
3. `labeler/index.html`을 Chrome으로 열어요.
4. 이름을 입력하고 **pilot 폴더**를 선택해요.
5. 키보드로 라벨링해요.
   - `1`~`9`: 동물상 선택 후 자동으로 다음 사진
   - `0`: 판단 불가
   - `←`/`→`: 이전/다음 사진
   - 진행 상황은 브라우저에 자동 저장돼요.
6. **CSV 내보내기**로 받은 `labels_이름.csv`를 드라이브의 `data/labels/`에 올려요.
7. 팀장이 `merge_labels.py --pilot-only`를 실행해 일치도(Fleiss' kappa)와 자주 헷갈린 쌍을 확인하고, 기준표를 보완해요.
   - kappa 0.2~0.4: "약한 일치". 주관적인 과제라 정상 범위예요.
   - 0.2 미만이면 기준표를 더 구체적으로 다듬어요.
8. 각자 **main 폴더**를 라벨링해요. 같은 페이지에서 폴더만 바꿔 선택하면 돼요.

### 3일차: soft label 만들기 + 데이터 분리
1. 전원의 CSV가 `data/labels/`에 모이면 `merge_labels.py`를 실행해요.
   - 투표 비율이 그대로 soft label이 돼요. 예: 강아지 2표 + 곰 1표 → 강아지 0.67, 곰 0.33
2. `split_dataset.py`를 실행해요.
   - 유사 이미지(dHash)를 같은 그룹으로 묶어요.
   - 클래스 비율을 유지하면서 70/15/15로 나눠요.

### 4일차: 학습 및 실험 (Colab GPU, 30~60분)
`train.py --exp all`을 실행하면 아래 9개 실험이 한 번에 돌아가요.

| 실험 | 설명 |
|---|---|
| `mlp_f100` | 강의(DL10)의 MLP baseline (64×64 이미지를 펼쳐서 입력) |
| `linear_f025~f100` | ResNet18 고정, 마지막 층만 학습 × 데이터 25/50/75/100% |
| `finetune_f025~f100` | ResNet18 전체 미세조정 × 데이터 25/50/75/100% |

- **손실 함수**: soft label 교차 엔트로피 `L = −Σ y_k · log softmax(z)_k`. DL07 softmax 회귀의 손실을 확률 분포 정답으로 일반화한 것이에요.
- **결과 파일** (`results/`)
  - `summary.csv`: 실험별 Accuracy, Precision, Recall, F1, vote-hit, KL
  - `datasize.png`: 데이터 규모에 따른 성능 그래프
  - `<실험>/curves.png`: Train/Validation Loss·Accuracy 곡선
  - `<실험>/confusion.png`: Confusion Matrix
- **평가 기준**
  - 정답은 최다 득표 클래스예요.
  - `vote_hit_rate`는 모델이 고른 클래스가 라벨러 중 1명 이상의 표를 받은 비율이에요. 정답이 하나로 정해지지 않는 과제라서 이 보조 지표도 같이 봐요.

### 5일차: ONNX 변환 + 웹 연결
1. `export_onnx.py`를 실행하면 `face_type.onnx`(약 45MB)와 `meta.json`이 생성돼요.
2. 두 파일을 저장소의 `web/model/`에 넣고 commit/push해요.
   - GitHub 파일 한도(100MB)보다 작아서 그대로 올릴 수 있어요.
3. **로컬에서 실행하기**: VS Code에서 `web/index.html`을 우클릭 → *Open with Live Server*. 또는:
   ```bash
   cd web && python -m http.server 8000   # → http://localhost:8000
   ```
   - 카메라는 `localhost`나 `https`에서만 동작해요. 파일을 더블클릭해서 열면 동작하지 않아요.
   - 모델 파일이 없으면 노란 배너와 함께 **가짜 결과**가 나와요. 그래서 학습이 끝나기 전에도 웹 작업을 진행할 수 있어요.

### 6일차: Render 배포 + Supabase(2단계)

**Render 배포**
1. render.com → New → **Static Site** → GitHub 저장소를 연결해요.
2. 설정값을 입력해요.
   - Root Directory: `web`
   - Build Command: `echo ok`
   - Publish Directory: `.`
3. push할 때마다 자동으로 재배포돼요.

**Supabase**
1. supabase.com에서 새 프로젝트를 만들어요.
2. **SQL Editor**에 `supabase/schema.sql`을 붙여넣고 Run해요.
3. Authentication → Sign In / Providers → Email에서 *Confirm email*을 끄면 데모할 때 편해요(메일 인증 생략).
4. Project Settings → API에서 **Project URL**과 **anon(publishable) key**를 복사해 `web/js/config.js`에 넣어요.
   - 이 키는 공개돼도 괜찮은 키예요. 데이터는 RLS 정책이 보호해요.
   - **service_role 키는 절대 넣으면 안 돼요.**
5. 이제 웹 오른쪽 위에 **로그인** 버튼이 나타나요.
6. 로그인한 뒤 **저장하고 실제 회원 중에서 찾기**를 누르면, 내 프로필과 확률 벡터가 저장되고 다른 회원 중에서 추천돼요.
   - 테스트할 때는 계정을 여러 개 만들어서 서로 매칭해 보면 돼요.

### 7일차: 정리
- 보고서와 발표에 `summary.csv`, `datasize.png`, confusion matrix, 파일럿 kappa를 넣어요.
- 데모 시나리오: 카메라 촬영 → 동물상 → 이상형 선택 → Top 5 추천

---

## 매칭 점수

```
fit(A→B) = (A의 이상형 벡터 · B의 얼굴 확률 벡터) / max(A의 이상형 벡터)
           (B가 A의 1순위 동물상 100%이면 1.0)
상호 점수 = (fit(A→B) + fit(B→A)) / 2           ← 양쪽 취향을 모두 반영
최종 점수 = 0.8 × 상호 점수 + 0.2 × 취미 자카드 유사도
조건: 성별·나이 조건이 서로 맞는 사람만 후보로 포함
```
이상형은 최대 3순위까지 고르고, 순위별로 가중치를 0.5 / 0.3 / 0.2로 줘요.
가중치는 `web/js/config.js`에서 바꿀 수 있어요.

## 꼭 지켜야 할 것

- **클래스 순서**: `training/common.py`, `web/js/config.js`, `labeler/index.html`의 클래스 순서가 같아야 해요. 바꾸려면 세 파일을 모두 고쳐요.
- **크롭 방식**: 학습과 웹에서 같은 크롭 수식을 써야 해요. `CROP_MARGIN`을 바꾸면 데이터 준비부터 다시 해야 해요.
- **윤리와 개인정보**
  - 동물상은 과학적 분류가 아니라 재미 요소예요. 발표할 때도 그렇게 소개하는 게 좋아요.
  - 사진은 저장하지 않아요.
  - FairFace는 CC BY 4.0 라이선스라 출처를 표기해야 해요.
