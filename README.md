# 🐾 동물상 소개팅 — 딥러닝 얼굴상 분류 기반 매칭 웹

얼굴 사진을 11가지 동물상(강아지·고양이·토끼·여우·곰·사슴·공룡·늑대·다람쥐·말·쥐)으로 분류하고,
서로의 이상형을 함께 반영한 **상호 매칭 점수**로 소개팅 상대 Top 5를 추천합니다.
한쪽의 취향만 보지 않고 서로 취향이 맞는 조합을 위로 올리는 방식으로, 데이팅 추천 연구에서는 이를 **상호 추천(reciprocal recommendation)**이라고 부릅니다.

- **모델**: ImageNet으로 사전학습된 ResNet18을 전이학습합니다(마지막 층만 학습 / 전체 미세조정 비교). 라벨러 투표 비율(soft label)로 학습해요.
- **추론**: 브라우저에서 실행합니다(MediaPipe 얼굴 검출 + ONNX Runtime Web). 사진은 서버로 전송되지 않아요.
- **화면 표시**: 동물상 하나만 보여 줍니다. 매칭 계산에는 11개 클래스의 확률 벡터 전체를 씁니다.
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
| 라벨링 | `labeler/index.html` (자체 도구: 키보드 단축키로 빠르게 라벨링, 설치 불필요) |
| 배포 | **Render Static Site** (잠들지 않음, HTTPS 자동 → 카메라 사용 가능) |
| 회원 / DB | **Supabase** (무료 Postgres + 로그인) |

- **라벨링 도구**: 별도 설치 없이 `labeler/index.html`을 더블클릭하면 바로 열려요.
- **웹**: 빌드 도구 없이 CDN으로 라이브러리를 불러와요. `npm install`도 빌드도 필요 없어요.
- **코드 저장소**: https://github.com/jhnxk/face-matching

---

## 일정별 진행 방법

| 일차 | 할 일 |
|---|---|
| 1일차 | GitHub 저장소·환경 세팅, 이미지 데이터 추출 + MediaPipe로 얼굴 자르기(Colab), 기준표 작성 |
| 2일차 | 파일럿 라벨링(50장) → 기준표 보완 → 본 라벨링 시작 |
| 3일차 | 본 라벨링 마무리 → soft label 합치기, train/val/test 분리 |
| 4일차 | 학습 및 실험(데이터 규모 4단계, 학습 방식 2종, MLP baseline) |
| 5일차 | ONNX 변환, 웹 1단계(카메라 → 분류 → 데모 매칭) |
| 6일차 | Render 배포, 2단계(Supabase 회원가입 + DB) |
| 7일차 | 버그 수정, 결과 그래프, 발표 자료 |

### 1일차: 저장소·환경 세팅 + 데이터 준비 + 기준표

**① GitHub 저장소·환경 세팅**
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

**② 이미지 데이터 추출 + 얼굴 자르기 (Colab)**
1. GitHub의 노트북 파일을 Colab에서 열어요(colab.research.google.com → GitHub 탭).
2. 런타임 유형을 **T4 GPU**로 바꿔요.
3. 셀을 위에서부터 실행해요.
   - `prepare_fairface.py`가 FairFace(Hugging Face `HuggingFaceM4/FairFace`, 1.25 버전) train 분할 86,744장 중 **FairFace가 제공하는 race·age 라벨**로 East Asian, 20~39세 후보(7,147장)를 고른 뒤, 성별별로 무작위 추출해 남녀 750장씩 총 1,500장을 만들어요.
   - 뽑은 사진은 MediaPipe로 **눈높이를 맞춰 정렬하고 224×224 크기로 잘라** 구글 드라이브에 저장해요. 얼굴 검출에 실패한 사진(135장)은 건너뛰고 다음 후보로 채웠어요.
   - `meta.csv`의 `fairface_index`는 FairFace train 분할의 원본 행 번호라서, 원본 사진과 라벨을 다시 확인할 수 있어요.
   - 마지막에 나오는 `FaceLandmarker.__del__ … TypeError`는 종료 시 정리 과정의 메시지라 무시해도 돼요.
   - 런타임이 끊기면 변수와 위치가 초기화되므로 **①드라이브 연결 셀 → ②git clone 셀**부터 다시 실행해요.
4. 크롭 결과 미리보기 셀로 얼굴이 제대로 잘렸는지 확인해요.

**③ 기준표 작성**
- 팀 회의에서 `labeler/index.html` 오른쪽의 **판단 기준표**(11종 설명)를 함께 다듬어요(파일 안의 `CLASSES` 배열).

### 2일차: 파일럿 라벨링 → 기준표 보완 → 본 라벨링 시작

**① 라벨러 배정**
1. Colab에서 `make_assignments.py`를 실행해요(팀원 이름은 노트북에서 수정).
   - 결과는 `data/labeling/<이름>/pilot`과 `.../main` 폴더로 나뉘어요.
   - 파일럿 50장은 5명 모두가 라벨링해요.
   - 나머지 1,450장은 장당 3명이 순환 배정으로 라벨링해요.
   - 1인당 **파일럿 50장 + 본 라벨링 870장 = 920장**(약 1시간)이에요.
2. 드라이브 `data/` 안에 결과를 받을 `labels_result/` 폴더를 만들고, `labeling/`과 함께 팀원에게 공유해요.

**② 파일럿 라벨링 50장**
1. 각자 드라이브에서 자기 폴더를 내려받아요.
2. `labeler/index.html`을 Chrome으로 열어요.
3. 이름을 **배정 때와 똑같이**(주환 / 정원 / 정유 / 동윤 / 채하) 입력하고 **pilot 폴더**를 선택해요.
4. 키보드로 라벨링해요.
   - `1`~`9`, `0`(말상), `-`(쥐상): 동물상 선택 후 자동으로 다음 사진
   - `X`: 판단 불가
   - `←`/`→`: 이전/다음 사진
   - 진행 상황은 브라우저에 자동 저장돼요. 같은 PC·같은 브라우저로 끝까지 진행해요.
   - 팀원끼리 상의하지 않고 각자 3~4초 안에 판단해요.
5. **CSV 내보내기**로 받은 `labels_이름.csv`를 드라이브의 `data/labels_result/`에 올려요.

**③ 기준표 보완**
1. 파일럿 결과를 분석해요.
   ```python
   !python training/merge_labels.py --data-dir $DATA --labels-glob "$DATA/labels_result/*.csv" --pilot-only
   ```
2. 일치도(Fleiss' kappa)와 자주 헷갈린 쌍을 보고 기준표를 보완해요.
   - kappa 0.2~0.4: "약한 일치". 주관적인 과제라 정상 범위예요.
   - 0.2 미만이면 기준표를 더 구체적으로 다듬어요.

**④ 본 라벨링 시작**
- 같은 라벨링 페이지에서 이름은 그대로 두고 **main 폴더**(870장)를 선택해 라벨링을 시작해요.

### 3일차: 본 라벨링 마무리 → soft label 합치기 + 데이터 분리
1. 각자 **main 폴더 라벨링을 마무리**하고, 다시 CSV를 내보내 `labels_result/`의 기존 파일에 덮어써요(파일럿 + 본 라벨링이 한 파일에 들어 있음).
2. 전원의 CSV가 모이면 팀장이 `merge_labels.py`를 실행해요.
   ```python
   !python training/merge_labels.py --data-dir $DATA --labels-glob "$DATA/labels_result/*.csv"
   ```
   - 투표 비율이 그대로 soft label이 돼요. 예: 강아지 2표 + 곰 1표 → 강아지 0.67, 곰 0.33
3. `split_dataset.py`를 실행해요.
   - 유사 이미지(dHash)를 같은 그룹으로 묶어요.
   - 클래스 비율을 유지하면서 70/15/15로 나눠요(약 1,050 / 225 / 225장).
   - 학습·검증·테스트 모두 **라벨링한 1,500장 안에서** 겹치지 않게 나눠요. 유효 표가 2개 미만인 사진은 제외돼요.
   - 모든 실험은 같은 Test 세트로 평가하고, Test는 모델 선택에 쓰지 않아요.

### 4일차: 학습 및 실험 (Colab GPU, 30~60분)
`train.py --exp all`을 실행하면 아래 9개 실험이 한 번에 돌아가요.

| 실험 | 설명 |
|---|---|
| `mlp_f100` | 강의(DL10)의 MLP baseline (64×64 이미지를 펼쳐서 입력) |
| `linear_f025~f100` | ResNet18 고정, 마지막 층만 학습 × 데이터 25/50/75/100% |
| `finetune_f025~f100` | ResNet18 전체 미세조정 × 데이터 25/50/75/100% |

- **데이터 규모 실험(4단계)**: Train에서만 25/50/75/100%를 클래스 비율을 유지하며 뽑아요. 작은 부분집합은 큰 부분집합에 포함돼요. Val/Test는 고정이에요.
- **학습 방식 비교(2종)**: 두 방식 모두 같은 ResNet18(18층)이며, 사전학습된 층을 고정하고 마지막 층만 학습하느냐 / 전체 층을 함께 학습하느냐의 차이예요.
- **손실 함수**: soft label 교차 엔트로피 `L = −Σ y_k · log softmax(z)_k`. DL07 softmax 회귀의 손실을 확률 분포 정답으로 일반화한 것이에요.
- **결과 파일** (`results/`)
  - `summary.csv`: 실험별 Accuracy, Precision, Recall, F1, vote-hit, KL
  - `datasize.png`: 데이터 규모에 따른 성능 그래프
  - `<실험>/curves.png`: Train/Validation Loss·Accuracy 곡선
  - `<실험>/confusion.png`: Confusion Matrix
- **평가 기준**
  - 정답은 최다 득표 클래스예요.
  - `vote_hit_rate`는 모델이 고른 클래스가 라벨러 중 1명 이상의 표를 받은 비율이에요. 정답이 하나로 정해지지 않는 과제라서 이 보조 지표도 같이 봐요.

### 5일차: ONNX 변환 + 웹 1단계(카메라 → 분류 → 데모 매칭)
1. `export_onnx.py`를 실행하면 `face_type.onnx`(약 45MB)와 `meta.json`이 생성돼요.
   - 기본값은 `finetune_f100` 모델이에요. 실험에서 다른 모델이 더 좋았다면 `--ckpt`로 그 모델을 지정해요.
2. 두 파일을 저장소의 `web/model/`에 넣고 commit/push해요.
   - GitHub 파일 한도(100MB)보다 작아서 그대로 올릴 수 있어요.
3. **로컬에서 웹 1단계 확인하기**: VS Code에서 `web/index.html`을 우클릭 → *Open with Live Server*. 또는:
   ```bash
   cd web && python -m http.server 8000   # → http://localhost:8000
   ```
   - 카메라 촬영(또는 사진 올리기) → 동물상 결과 → 이상형 선택 → **데모 후보 40명 중 Top 5**까지 동작하는지 확인해요.
   - 파일을 더블클릭(`file://`)해서 열면 JavaScript 모듈이 차단돼 버튼이 활성화되지 않아요. 반드시 Live Server나 배포 주소로 열어요.
   - 카메라는 `localhost`나 `https`에서만 동작해요.
   - 모델 파일이 없으면 노란 배너와 함께 **가짜 결과**가 나와요. 그래서 학습이 끝나기 전에도 웹 작업을 진행할 수 있어요.

### 6일차: Render 배포 + 2단계(Supabase 회원가입 + DB)

**Render 배포**
1. render.com → New → **Static Site** → GitHub 저장소를 연결해요.
2. 설정값을 입력해요.
   - Root Directory: `web`
   - Build Command: `echo ok`
   - Publish Directory: `.`
3. push할 때마다 자동으로 재배포돼요.
4. 배포 주소(`https://…onrender.com`)를 휴대폰으로 열어 카메라 촬영까지 확인해요.

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

### 7일차: 버그 수정, 결과 그래프, 발표 자료
- **버그 수정**: 배포 주소에서 PC·휴대폰으로 전체 흐름을 테스트하고, 브라우저 콘솔(F12) 오류를 확인해 고쳐요.
- **결과 그래프**: `summary.csv`, `datasize.png`, 실험별 `curves.png`·`confusion.png`, 파일럿 kappa를 정리해요.
- **발표 자료**: 데모 시나리오(카메라 촬영 → 동물상 → 이상형 선택 → Top 5 추천)와 실험 결과를 정리해요.

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
  - 인종(East Asian) 선별은 FairFace가 제공한 라벨을 그대로 사용했고, 우리 모델은 동물상만 분류해요.
