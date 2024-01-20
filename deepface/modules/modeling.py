# built-in dependencies
from typing import Any

# project dependencies
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    FbDeepFace,
    DeepID,
    DlibResNet,
    ArcFace,
    SFace,
)
from deepface.extendedmodels import Age, Gender, Race, Emotion


def build_model(model_name: str) -> Any:
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built model class
    """

    # singleton design pattern
    global model_obj

    models = {
        "VGG-Face": VGGFace.VggFace,
        "OpenFace": OpenFace.OpenFace,
        "Facenet": Facenet.FaceNet128d,
        "Facenet512": Facenet.FaceNet512d,
        "DeepFace": FbDeepFace.DeepFace,
        "DeepID": DeepID.DeepId,
        "Dlib": DlibResNet.Dlib,
        "ArcFace": ArcFace.ArcFace,
        "SFace": SFace.SFace,
        "Emotion": Emotion.FacialExpression,
        "Age": Age.ApparentAge,
        "Gender": Gender.Gender,
        "Race": Race.Race,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]
