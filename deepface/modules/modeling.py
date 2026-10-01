from __future__ import annotations

# built-in dependencies
import importlib
from typing import TYPE_CHECKING, Any, Dict, Final, TypedDict

# project dependencies
from deepface.commons import backend_utils
from deepface.commons.backend_utils import ONNX, PYTORCH, TENSORFLOW
from deepface.modules.exceptions import UnimplementedError

# Models are not imported here but built lazily, on demand. Importing them eagerly would
# import tensorflow, pytorch and onnxruntime all, whereas deepface runs on one of them - see
# deepface.commons.backend_utils for the way the backend engine is picked.

if TYPE_CHECKING:
    cached_models: Dict[str, Dict[str, Any]] = {}

# a model that does not depend on a backend engine, e.g. an opencv or an onnx one,
# is registered with this key and is available whichever engine deepface runs on
ANY = "any"


# where the implementations of a model live, keyed by the backend engine or by ANY
ModelSources = Dict[str, str]


class AvailableModels(TypedDict):
    facial_recognition: Dict[str, ModelSources]
    spoofing: Dict[str, ModelSources]
    facial_attribute: Dict[str, ModelSources]
    face_detector: Dict[str, ModelSources]


AVAILABLE_MODELS: Final[AvailableModels] = {
    "facial_recognition": {
        "VGG-Face": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.VGGFace.VggFaceClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.VGGFace.VggFaceClient",
            ONNX: "deepface.models.facial_recognition.onnx.VGGFace.VggFaceClient",
        },
        "OpenFace": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.OpenFace.OpenFaceClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.OpenFace.OpenFaceClient",
            ONNX: "deepface.models.facial_recognition.onnx.OpenFace.OpenFaceClient",
        },
        "Facenet": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.Facenet.FaceNet128dClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.Facenet.FaceNet128dClient",
            ONNX: "deepface.models.facial_recognition.onnx.Facenet.FaceNet128dClient",
        },
        "Facenet512": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.Facenet.FaceNet512dClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.Facenet.FaceNet512dClient",
            ONNX: "deepface.models.facial_recognition.onnx.Facenet.FaceNet512dClient",
        },
        "DeepFace": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.FbDeepFace.DeepFaceClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.FbDeepFace.DeepFaceClient",
            ONNX: "deepface.models.facial_recognition.onnx.FbDeepFace.DeepFaceClient",
        },
        "DeepID": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.DeepID.DeepIdClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.DeepID.DeepIdClient",
            ONNX: "deepface.models.facial_recognition.onnx.DeepID.DeepIdClient",
        },
        "Dlib": {ANY: "deepface.models.facial_recognition.dlib.Dlib.DlibClient"},
        "ArcFace": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.ArcFace.ArcFaceClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.ArcFace.ArcFaceClient",
            ONNX: "deepface.models.facial_recognition.onnx.ArcFace.ArcFaceClient",
        },
        "SFace": {ANY: "deepface.models.facial_recognition.onnx.SFace.SFaceClient"},
        "GhostFaceNet": {
            TENSORFLOW: "deepface.models.facial_recognition.tf.GhostFaceNet.GhostFaceNetClient",
            PYTORCH: "deepface.models.facial_recognition.pytorch.GhostFaceNet.GhostFaceNetClient",
            ONNX: "deepface.models.facial_recognition.onnx.GhostFaceNet.GhostFaceNetClient",
        },
        "Buffalo_L": {ANY: "deepface.models.facial_recognition.onnx.Buffalo_L.Buffalo_L"},
    },
    "spoofing": {
        "Fasnet": {
            TENSORFLOW: "deepface.models.spoofing.tf.FasNet.Fasnet",
            PYTORCH: "deepface.models.spoofing.pytorch.FasNet.Fasnet",
            ONNX: "deepface.models.spoofing.onnx.FasNet.Fasnet",
        },
    },
    "facial_attribute": {
        "Emotion": {
            TENSORFLOW: "deepface.models.demography.tf.Emotion.EmotionClient",
            PYTORCH: "deepface.models.demography.pytorch.Emotion.EmotionClient",
            ONNX: "deepface.models.demography.onnx.Emotion.EmotionClient",
        },
        "Age": {
            TENSORFLOW: "deepface.models.demography.tf.Age.ApparentAgeClient",
            PYTORCH: "deepface.models.demography.pytorch.Age.ApparentAgeClient",
            ONNX: "deepface.models.demography.onnx.Age.ApparentAgeClient",
        },
        "Gender": {
            TENSORFLOW: "deepface.models.demography.tf.Gender.GenderClient",
            PYTORCH: "deepface.models.demography.pytorch.Gender.GenderClient",
            ONNX: "deepface.models.demography.onnx.Gender.GenderClient",
        },
        "Race": {
            TENSORFLOW: "deepface.models.demography.tf.Race.RaceClient",
            PYTORCH: "deepface.models.demography.pytorch.Race.RaceClient",
            ONNX: "deepface.models.demography.onnx.Race.RaceClient",
        },
    },
    # face detectors bring their own dependencies - opencv, dlib, mediapipe, ultralytics
    # or retina-face - so they are offered whichever backend engine deepface runs on
    "face_detector": {
        "opencv": {ANY: "deepface.models.face_detection.OpenCv.OpenCvClient"},
        "mtcnn": {ANY: "deepface.models.face_detection.MtCnn.MtCnnClient"},
        "ssd": {ANY: "deepface.models.face_detection.Ssd.SsdClient"},
        "dlib": {ANY: "deepface.models.face_detection.Dlib.DlibClient"},
        "retinaface": {ANY: "deepface.models.face_detection.RetinaFace.RetinaFaceClient"},
        "mediapipe": {ANY: "deepface.models.face_detection.MediaPipe.MediaPipeClient"},
        "yolov8n": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV8n"},
        "yolov8m": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV8m"},
        "yolov8l": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV8l"},
        "yolov11n": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV11n"},
        "yolov11s": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV11s"},
        "yolov11m": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV11m"},
        "yolov11l": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV11l"},
        "yolov12n": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV12n"},
        "yolov12s": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV12s"},
        "yolov12m": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV12m"},
        "yolov12l": {ANY: "deepface.models.face_detection.Yolo.YoloDetectorClientV12l"},
        "yunet": {ANY: "deepface.models.face_detection.YuNet.YuNetClient"},
        "fastmtcnn": {ANY: "deepface.models.face_detection.FastMtCnn.FastMtCnnClient"},
        "centerface": {ANY: "deepface.models.face_detection.CenterFace.CenterFaceClient"},
    },
}


def get_model_class(task: str, model_name: str) -> type:
    """
    Find the class implementing a model for the backend engine in use, importing its
    module only when it is asked for
    Args:
        task (str): facial_recognition, facial_attribute, face_detector or spoofing
        model_name (str): model identifier
    Returns:
        model_class (type): class of the model, not an instance of it
    """
    if task not in AVAILABLE_MODELS.keys():
        raise UnimplementedError(f"unimplemented task - {task}")

    sources = AVAILABLE_MODELS[task].get(model_name)  # type: ignore[literal-required]
    if not sources:
        raise UnimplementedError(f"Invalid model_name passed - {task}/{model_name}")

    backend = backend_utils.get_backend_engine()
    source = sources.get(backend) or sources.get(ANY)

    if not source:
        available = [engine for engine in sources if engine != ANY]
        raise UnimplementedError(
            f"{task}/{model_name} is not implemented for the {backend} backend engine. "
            f"It is available for {available}. You may switch the backend engine with the "
            f"{backend_utils.BACKEND_ENGINE_ENV_VAR} environment variable."
        )

    module_path, _, class_name = source.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


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
        raise UnimplementedError(f"unimplemented task - {task}")

    if "cached_models" not in globals():
        cached_models = {current_task: {} for current_task in AVAILABLE_MODELS.keys()}

    # pylint: disable=possibly-used-before-assignment
    if cached_models[task].get(model_name) is None:
        cached_models[task][model_name] = get_model_class(task=task, model_name=model_name)()

    return cached_models[task][model_name]
