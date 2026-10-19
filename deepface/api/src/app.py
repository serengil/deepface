# 3rd parth dependencies
try:
    from flask import Flask
    from flask_cors import CORS
    from dotenv import load_dotenv
except ImportError as import_err:
    raise ImportError(
        f"{import_err.name} is required to run the deepface api. "
        'Please run `pip install "deepface[api]"` to install it.'
    ) from import_err

# load environment variables from .env first things first
load_dotenv()


# pylint: disable=wrong-import-position
# project dependencies
from deepface import __version__
from deepface.api.src.modules.core.routes import blueprint
from deepface.api.src.dependencies.variables import Variables
from deepface.api.src.dependencies.container import Container
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

    container.model_service.load_models_on_startup()

    app.register_blueprint(blueprint)

    logger.info(f"Welcome to DeepFace API v{__version__}!")
    return app
