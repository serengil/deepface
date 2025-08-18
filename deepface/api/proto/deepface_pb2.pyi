from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Models(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VGG_FACE: _ClassVar[Models]
    FACENET: _ClassVar[Models]
    FACENET512: _ClassVar[Models]
    OPENFACE: _ClassVar[Models]
    DEEPFACE: _ClassVar[Models]
    DEEPID: _ClassVar[Models]
    ARCFACE: _ClassVar[Models]
    DLIB_MODEL: _ClassVar[Models]
    SFACE: _ClassVar[Models]
    GHOSTFACENET: _ClassVar[Models]
    BUFFALO_L: _ClassVar[Models]

class Detectors(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPENCV: _ClassVar[Detectors]
    SSD: _ClassVar[Detectors]
    DLIB: _ClassVar[Detectors]
    MTCNN: _ClassVar[Detectors]
    FASTMTCNN: _ClassVar[Detectors]
    RETINAFACE: _ClassVar[Detectors]
    MEDIAPIPE: _ClassVar[Detectors]
    YOLOV8: _ClassVar[Detectors]
    YOLOV11S: _ClassVar[Detectors]
    YOLOV11N: _ClassVar[Detectors]
    YOLOV11M: _ClassVar[Detectors]
    YUNET: _ClassVar[Detectors]
    CENTERFACE: _ClassVar[Detectors]

class DistanceMetrics(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COSINE: _ClassVar[DistanceMetrics]
    EUCLIDEAN: _ClassVar[DistanceMetrics]
    EUCLIDEAN_L2: _ClassVar[DistanceMetrics]
    ANGULAR: _ClassVar[DistanceMetrics]
VGG_FACE: Models
FACENET: Models
FACENET512: Models
OPENFACE: Models
DEEPFACE: Models
DEEPID: Models
ARCFACE: Models
DLIB_MODEL: Models
SFACE: Models
GHOSTFACENET: Models
BUFFALO_L: Models
OPENCV: Detectors
SSD: Detectors
DLIB: Detectors
MTCNN: Detectors
FASTMTCNN: Detectors
RETINAFACE: Detectors
MEDIAPIPE: Detectors
YOLOV8: Detectors
YOLOV11S: Detectors
YOLOV11N: Detectors
YOLOV11M: Detectors
YUNET: Detectors
CENTERFACE: Detectors
COSINE: DistanceMetrics
EUCLIDEAN: DistanceMetrics
EUCLIDEAN_L2: DistanceMetrics
ANGULAR: DistanceMetrics

class FacialArea(_message.Message):
    __slots__ = ("left_eye", "right_eye", "mouth_left", "mouth_right", "nose", "h", "w", "x", "y")
    LEFT_EYE_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EYE_FIELD_NUMBER: _ClassVar[int]
    MOUTH_LEFT_FIELD_NUMBER: _ClassVar[int]
    MOUTH_RIGHT_FIELD_NUMBER: _ClassVar[int]
    NOSE_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    left_eye: _containers.RepeatedScalarFieldContainer[int]
    right_eye: _containers.RepeatedScalarFieldContainer[int]
    mouth_left: _containers.RepeatedScalarFieldContainer[int]
    mouth_right: _containers.RepeatedScalarFieldContainer[int]
    nose: _containers.RepeatedScalarFieldContainer[int]
    h: int
    w: int
    x: int
    y: int
    def __init__(self, left_eye: _Optional[_Iterable[int]] = ..., right_eye: _Optional[_Iterable[int]] = ..., mouth_left: _Optional[_Iterable[int]] = ..., mouth_right: _Optional[_Iterable[int]] = ..., nose: _Optional[_Iterable[int]] = ..., h: _Optional[int] = ..., w: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class AnalyzeRequest(_message.Message):
    __slots__ = ("image_url", "actions", "detector_backend", "enforce_detection", "align", "anti_spoofing", "max_faces")
    class Action(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        AGE: _ClassVar[AnalyzeRequest.Action]
        GENDER: _ClassVar[AnalyzeRequest.Action]
        EMOTION: _ClassVar[AnalyzeRequest.Action]
        RACE: _ClassVar[AnalyzeRequest.Action]
    AGE: AnalyzeRequest.Action
    GENDER: AnalyzeRequest.Action
    EMOTION: AnalyzeRequest.Action
    RACE: AnalyzeRequest.Action
    IMAGE_URL_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    MAX_FACES_FIELD_NUMBER: _ClassVar[int]
    image_url: str
    actions: _containers.RepeatedScalarFieldContainer[AnalyzeRequest.Action]
    detector_backend: Detectors
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    max_faces: int
    def __init__(self, image_url: _Optional[str] = ..., actions: _Optional[_Iterable[_Union[AnalyzeRequest.Action, str]]] = ..., detector_backend: _Optional[_Union[Detectors, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ..., max_faces: _Optional[int] = ...) -> None: ...

class AnalyzeResponse(_message.Message):
    __slots__ = ("results",)
    class Emotion(_message.Message):
        __slots__ = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
        ANGRY_FIELD_NUMBER: _ClassVar[int]
        DISGUST_FIELD_NUMBER: _ClassVar[int]
        FEAR_FIELD_NUMBER: _ClassVar[int]
        HAPPY_FIELD_NUMBER: _ClassVar[int]
        NEUTRAL_FIELD_NUMBER: _ClassVar[int]
        SAD_FIELD_NUMBER: _ClassVar[int]
        SURPRISE_FIELD_NUMBER: _ClassVar[int]
        angry: float
        disgust: float
        fear: float
        happy: float
        neutral: float
        sad: float
        surprise: float
        def __init__(self, angry: _Optional[float] = ..., disgust: _Optional[float] = ..., fear: _Optional[float] = ..., happy: _Optional[float] = ..., neutral: _Optional[float] = ..., sad: _Optional[float] = ..., surprise: _Optional[float] = ...) -> None: ...
    class Gender(_message.Message):
        __slots__ = ("man", "woman")
        MAN_FIELD_NUMBER: _ClassVar[int]
        WOMAN_FIELD_NUMBER: _ClassVar[int]
        man: float
        woman: float
        def __init__(self, man: _Optional[float] = ..., woman: _Optional[float] = ...) -> None: ...
    class Race(_message.Message):
        __slots__ = ("asian", "black", "indian", "latino_hispanic", "middle_eastern", "white")
        ASIAN_FIELD_NUMBER: _ClassVar[int]
        BLACK_FIELD_NUMBER: _ClassVar[int]
        INDIAN_FIELD_NUMBER: _ClassVar[int]
        LATINO_HISPANIC_FIELD_NUMBER: _ClassVar[int]
        MIDDLE_EASTERN_FIELD_NUMBER: _ClassVar[int]
        WHITE_FIELD_NUMBER: _ClassVar[int]
        asian: float
        black: float
        indian: float
        latino_hispanic: float
        middle_eastern: float
        white: float
        def __init__(self, asian: _Optional[float] = ..., black: _Optional[float] = ..., indian: _Optional[float] = ..., latino_hispanic: _Optional[float] = ..., middle_eastern: _Optional[float] = ..., white: _Optional[float] = ...) -> None: ...
    class Result(_message.Message):
        __slots__ = ("age", "dominant_emotion", "dominant_gender", "dominant_race", "face_confidence", "emotion", "gender", "race", "facial_area")
        AGE_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_EMOTION_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_GENDER_FIELD_NUMBER: _ClassVar[int]
        DOMINANT_RACE_FIELD_NUMBER: _ClassVar[int]
        FACE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
        EMOTION_FIELD_NUMBER: _ClassVar[int]
        GENDER_FIELD_NUMBER: _ClassVar[int]
        RACE_FIELD_NUMBER: _ClassVar[int]
        FACIAL_AREA_FIELD_NUMBER: _ClassVar[int]
        age: int
        dominant_emotion: str
        dominant_gender: str
        dominant_race: str
        face_confidence: float
        emotion: AnalyzeResponse.Emotion
        gender: AnalyzeResponse.Gender
        race: AnalyzeResponse.Race
        facial_area: FacialArea
        def __init__(self, age: _Optional[int] = ..., dominant_emotion: _Optional[str] = ..., dominant_gender: _Optional[str] = ..., dominant_race: _Optional[str] = ..., face_confidence: _Optional[float] = ..., emotion: _Optional[_Union[AnalyzeResponse.Emotion, _Mapping]] = ..., gender: _Optional[_Union[AnalyzeResponse.Gender, _Mapping]] = ..., race: _Optional[_Union[AnalyzeResponse.Race, _Mapping]] = ..., facial_area: _Optional[_Union[FacialArea, _Mapping]] = ...) -> None: ...
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[AnalyzeResponse.Result]
    def __init__(self, results: _Optional[_Iterable[_Union[AnalyzeResponse.Result, _Mapping]]] = ...) -> None: ...

class RepresentRequest(_message.Message):
    __slots__ = ("image_url", "model_name", "detector_backend", "enforce_detection", "align", "anti_spoofing", "max_faces")
    IMAGE_URL_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    MAX_FACES_FIELD_NUMBER: _ClassVar[int]
    image_url: str
    model_name: Models
    detector_backend: Detectors
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    max_faces: int
    def __init__(self, image_url: _Optional[str] = ..., model_name: _Optional[_Union[Models, str]] = ..., detector_backend: _Optional[_Union[Detectors, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ..., max_faces: _Optional[int] = ...) -> None: ...

class RepresentResponse(_message.Message):
    __slots__ = ("results",)
    class Results(_message.Message):
        __slots__ = ("embedding", "face_confidence", "facial_area")
        EMBEDDING_FIELD_NUMBER: _ClassVar[int]
        FACE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
        FACIAL_AREA_FIELD_NUMBER: _ClassVar[int]
        embedding: _containers.RepeatedScalarFieldContainer[float]
        face_confidence: float
        facial_area: FacialArea
        def __init__(self, embedding: _Optional[_Iterable[float]] = ..., face_confidence: _Optional[float] = ..., facial_area: _Optional[_Union[FacialArea, _Mapping]] = ...) -> None: ...
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[RepresentResponse.Results]
    def __init__(self, results: _Optional[_Iterable[_Union[RepresentResponse.Results, _Mapping]]] = ...) -> None: ...

class VerifyRequest(_message.Message):
    __slots__ = ("image1_url", "image2_url", "model_name", "detector_backend", "distance_metric", "enforce_detection", "align", "anti_spoofing")
    IMAGE1_URL_FIELD_NUMBER: _ClassVar[int]
    IMAGE2_URL_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    ENFORCE_DETECTION_FIELD_NUMBER: _ClassVar[int]
    ALIGN_FIELD_NUMBER: _ClassVar[int]
    ANTI_SPOOFING_FIELD_NUMBER: _ClassVar[int]
    image1_url: str
    image2_url: str
    model_name: Models
    detector_backend: Detectors
    distance_metric: DistanceMetrics
    enforce_detection: bool
    align: bool
    anti_spoofing: bool
    def __init__(self, image1_url: _Optional[str] = ..., image2_url: _Optional[str] = ..., model_name: _Optional[_Union[Models, str]] = ..., detector_backend: _Optional[_Union[Detectors, str]] = ..., distance_metric: _Optional[_Union[DistanceMetrics, str]] = ..., enforce_detection: bool = ..., align: bool = ..., anti_spoofing: bool = ...) -> None: ...

class VerifyResponse(_message.Message):
    __slots__ = ("verified", "detector_backend", "model", "similarity_metric", "facial_areas", "distance", "threshold", "time")
    class FacialAreas(_message.Message):
        __slots__ = ("img1", "img2")
        IMG1_FIELD_NUMBER: _ClassVar[int]
        IMG2_FIELD_NUMBER: _ClassVar[int]
        img1: FacialArea
        img2: FacialArea
        def __init__(self, img1: _Optional[_Union[FacialArea, _Mapping]] = ..., img2: _Optional[_Union[FacialArea, _Mapping]] = ...) -> None: ...
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_METRIC_FIELD_NUMBER: _ClassVar[int]
    FACIAL_AREAS_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    verified: bool
    detector_backend: Detectors
    model: Models
    similarity_metric: DistanceMetrics
    facial_areas: VerifyResponse.FacialAreas
    distance: float
    threshold: float
    time: float
    def __init__(self, verified: bool = ..., detector_backend: _Optional[_Union[Detectors, str]] = ..., model: _Optional[_Union[Models, str]] = ..., similarity_metric: _Optional[_Union[DistanceMetrics, str]] = ..., facial_areas: _Optional[_Union[VerifyResponse.FacialAreas, _Mapping]] = ..., distance: _Optional[float] = ..., threshold: _Optional[float] = ..., time: _Optional[float] = ...) -> None: ...
