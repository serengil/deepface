from flask import Blueprint, request
from modules.core import service
from deepface.commons.logger import Logger

logger = Logger(module="api/src/routes.py")

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return "<h1>Welcome to DeepFace API!</h1>"


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(demographies)

    return demographies
