# 3rd parth dependencies
from flask import Flask
from deepface import DeepFace
from deepface.api.src.modules.core.routes import blueprint
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
