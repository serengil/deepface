from enum import Enum


class YoloModel(Enum):
    V8N = 0
    V11N = 1
    V11S = 2
    V11M = 3


# Model's weights paths
WEIGHT_NAMES = ["yolov8n-face.pt",
                "yolov11n-face.pt",
                "yolov11s-face.pt",
                "yolov11m-face.pt"]

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URLS = ["https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"]
