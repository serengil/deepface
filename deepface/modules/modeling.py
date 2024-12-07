# built-in dependencies
from typing import Any

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
)
from deepface.models.face_detection import (
    FastMtCnn,
    MediaPipe,
    MtCnn,
    OpenCv,
    Dlib as DlibDetector,
    RetinaFace,
    Ssd,
    Yolo,
    YuNet,
    CenterFace,
)
from deepface.models.demography import Age, Gender, Race, Emotion
from deepface.models.spoofing import FasNet


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace, GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, 'yolov11n', 'yolov11m', yunet,
                fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    models = {
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
            "yolov8": Yolo.YoloClientV8n,
            "yolov11n": Yolo.YoloClientV11n,
            "yolov11m": Yolo.YoloClientV11m,
            "yunet": YuNet.YuNetClient,
            "fastmtcnn": FastMtCnn.FastMtCnnClient,
            "centerface": CenterFace.CenterFaceClient,
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
