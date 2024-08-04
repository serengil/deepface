# built-in dependencies
from typing import Any

# project dependencies
from deepface.basemodels import (
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
from deepface.detectors import (
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
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.spoofmodels import FasNet


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace, GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, yunet,
                fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    tasks = ["facial_recognition", "spoofing", "facial_attribute", "face_detector"]

    if task not in tasks:
        raise ValueError(f"unimplemented task - {task}")

    # singleton design pattern
    global model_obj

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
            "yolov8": Yolo.YoloClient,
            "yunet": YuNet.YuNetClient,
            "fastmtcnn": FastMtCnn.FastMtCnnClient,
            "centerface": CenterFace.CenterFaceClient,
        },
    }

    if not "model_obj" in globals():
        model_obj = {current_task: {} for current_task in tasks}

    if not model_name in model_obj[task].keys():
        model = models[task].get(model_name)
        if model:
            model_obj[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return model_obj[task][model_name]
