from __future__ import annotations

# built-in dependencies
from typing import TYPE_CHECKING, Any, Final, TypedDict

# project dependencies
from deepface.models.facial_recognition import (
    VGGFace,
    OpenFace,
    FbDeepFace,
    DeepID,
    ArcFace,
    SFace,
    Dlib,
    Facenet,
    GhostFaceNet,
    Buffalo_L,
)
from deepface.models.face_detection import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib as DlibDetector,
    RetinaFace,
    Ssd,
    Yolo as YoloFaceDetector,
    YuNet,
    CenterFace,
)
from deepface.models.demography import Age, Gender, Race, Emotion
from deepface.models.spoofing import FasNet

if TYPE_CHECKING:
    from deepface.models.Demography import Demography
    from deepface.models.Detector import Detector
    from deepface.models.FacialRecognition import FacialRecognition


class AvailableModels(TypedDict):
    facial_recognition: dict[str, type[FacialRecognition]]
    spoofing: dict[str, type[FasNet.Fasnet]]
    facial_attribute: dict[str, type[Demography]]
    face_detector: dict[str, type[Detector]]


AVAILABLE_MODELS: Final[AvailableModels] = {
    "facial_recognition": {
        "VGG-Face": VGGFace.VggFaceClient,
        "OpenFace": OpenFace.OpenFaceClient,
        "Facenet": Facenet.FaceNet128dClient,
        "Facenet512": Facenet.FaceNet512dClient,
        "DeepFace": FbDeepFace.DeepFaceClient,
        "DeepID": DeepID.DeepIdClient,
        "Dlib": Dlib.DlibClient,
        "ArcFace": ArcFace.ArcFaceClient,
        "SFace": SFace.SFaceClient,
        "GhostFaceNet": GhostFaceNet.GhostFaceNetClient,
        "Buffalo_L": Buffalo_L.Buffalo_L,
    },
    "spoofing": {
        "Fasnet": FasNet.Fasnet,
    },
    "facial_attribute": {
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient,
    },
    "face_detector": {
        "opencv": OpenCv.OpenCvClient,
        "mtcnn": MtCnn.MtCnnClient,
        "ssd": Ssd.SsdClient,
        "dlib": DlibDetector.DlibClient,
        "retinaface": RetinaFace.RetinaFaceClient,
        "mediapipe": MediaPipe.MediaPipeClient,
        "yolov8": YoloFaceDetector.YoloDetectorClientV8n,
        "yolov11n": YoloFaceDetector.YoloDetectorClientV11n,
        "yolov11s": YoloFaceDetector.YoloDetectorClientV11s,
        "yolov11m": YoloFaceDetector.YoloDetectorClientV11m,
        "yunet": YuNet.YuNetClient,
        "fastmtcnn": FastMtCnn.FastMtCnnClient,
        "centerface": CenterFace.CenterFaceClient,
    },
}


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace and GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, 'yolov11n',
                'yolov11s', 'yolov11m', yunet, fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    if task not in AVAILABLE_MODELS.keys():
        raise ValueError(f"unimplemented task - {task}")

    if "cached_models" not in globals():
        cached_models = {current_task: {} for current_task in AVAILABLE_MODELS.keys()}

    if cached_models[task].get(model_name) is None:
        model = AVAILABLE_MODELS[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
