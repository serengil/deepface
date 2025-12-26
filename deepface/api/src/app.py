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
from deepface.commons.logger import Logger

logger = Logger()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # inject variables
    variables = Variables()
    blueprint.variables = variables  # type: ignore[attr-defined]

    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{__version__}!")
    return app
