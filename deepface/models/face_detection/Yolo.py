# built-in dependencies
import os
from typing import List, Any
from enum import Enum

# 3rd party dependencies
from numpy.typing import NDArray

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.commons import weight_utils

logger = Logger()


class YoloModel(Enum):
    # ref repo for v8n is https://github.com/derronqi/yolov8-face
    V8N = ("yolov8n-face.pt", "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb")
    V8M = (
        "yolov8m-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov8m-face.pt",
    )
    V8L = (
        "yolov8l-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov8l-face.pt",
    )
    V11N = (
        "yolov11n-face.pt",
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
    )
    V11S = (
        "yolov11s-face.pt",
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
    )
    V11M = (
        "yolov11m-face.pt",
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt",
    )
    V11L = (
        "yolov11l-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11l-face.pt",
    )
    V12N = (
        "yolov12n-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt",
    )
    V12S = (
        "yolov12s-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12s-face.pt",
    )
    V12M = (
        "yolov12m-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12m-face.pt",
    )
    V12L = (
        "yolov12l-face.pt",
        "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt",
    )

    def __init__(self, file_name: str, url: str):
        self.file_name = file_name
        self.url = url


class YoloDetectorClient(Detector):
    def __init__(self, model: YoloModel):
        super().__init__()
        self.model = self.build_model(model)

    def build_model(self, model: YoloModel) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the optional Ultralytics YOLO model
        try:
            import ultralytics
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. "
                "Please install using 'pip install ultralytics'"
            ) from e

        current_version = tuple(map(int, ultralytics.__version__.split(".")[:3]))

        if model.name == "V8N" and current_version < (8, 1, 6):
            raise ValueError("Yolov8n-face model requires ultralytics version 8.1.6 or higher.")
        if model.name in ["V11N", "V11S", "V11M"] and current_version < (8, 3, 203):
            raise ValueError("Yolov11 models require ultralytics version 8.3.203 or higher.")

        file_name, weight_url = model.value
        weight_file = weight_utils.download_weights_if_necessary(
            file_name=file_name, source_url=weight_url
        )

        # Return face_detector
        return YOLO(weight_file)

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # Detect faces
        results = self.model.predict(
            img,
            verbose=False,
            show=False,
            conf=float(os.getenv("YOLO_MIN_DETECTION_CONFIDENCE", "0.25")),
        )[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:

            if result.boxes is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            right_eye = None
            left_eye = None

            # yolo-facev8 is detecting eyes through keypoints,
            # while for v11 keypoints are always None
            if result.keypoints is not None:
                # right_eye_conf = result.keypoints.conf[0][0]
                # left_eye_conf = result.keypoints.conf[0][1]
                right_eye = result.keypoints.xy[0][0].tolist()
                left_eye = result.keypoints.xy[0][1].tolist()

                # eyes are list of float, need to cast them tuple of int
                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            resp.append(facial_area)

        return resp


class YoloDetectorClientV8n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV8m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8M)


class YoloDetectorClientV8l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V8L)


class YoloDetectorClientV11n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11M)


class YoloDetectorClientV11l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V11L)


class YoloDetectorClientV12n(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12N)


class YoloDetectorClientV12s(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12S)


class YoloDetectorClientV12m(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12M)


class YoloDetectorClientV12l(YoloDetectorClient):
    def __init__(self) -> None:
        super().__init__(YoloModel.V12L)
