# 3rd parth dependencies
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# load environment variables from .env first things first
load_dotenv()


# pylint: disable=wrong-import-position
# project dependencies
from deepface import __version__
from deepface.api.src.modules.core.routes import blueprint
from deepface.api.src.dependencies.variables import Variables
from deepface.api.src.dependencies.container import Container
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    variables = Variables()
    container = Container(variables=variables)

    # inject variables
    blueprint.variables = variables  # type: ignore[attr-defined]
    blueprint.container = container  # type: ignore[attr-defined]

    load_models_on_startup(variables)

    app.register_blueprint(blueprint)

    logger.info(f"Welcome to DeepFace API v{__version__}!")
    return app


def load_models_on_startup(variables: Variables) -> None:
    """Load models on startup to reduce latency on first request."""
    face_recognition_models = variables.face_recognition_models
    if face_recognition_models is not None:
        for model in face_recognition_models.split(","):
            DeepFace.build_model(task="facial_recognition", model_name=model.strip())
            logger.info(f"Facial Recognition Model {model} loaded on startup.")
    face_detection_models = variables.face_detection_models
    if face_detection_models is not None:
        for model in face_detection_models.split(","):
            DeepFace.build_model(task="face_detector", model_name=model.strip())
            logger.info(f"Face Detector Model {model} loaded on startup.")
