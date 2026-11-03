"""프로젝트 공통 상수. 웹(web/js/config.js)과 반드시 같은 순서를 유지해야 한다."""
from pathlib import Path

# 클래스 순서 = 모델 출력 인덱스 = 라벨링 단축키(1~9)
CLASSES = ["dog", "cat", "rabbit", "fox", "bear", "deer", "dino", "wolf", "squirrel"]
CLASSES_KO = ["강아지상", "고양이상", "토끼상", "여우상", "곰상", "사슴상", "공룡상", "늑대상", "다람쥐상"]
NUM_CLASSES = len(CLASSES)

# 얼굴 크롭/정규화 설정 (web/js/facecrop.js, web/js/classifier.js 와 동일해야 함)
IMG_SIZE = 224
CROP_MARGIN = 1.3                      # 랜드마크 bbox 대비 크롭 한 변 배율
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
