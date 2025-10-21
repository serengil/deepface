# 3rd parth dependencies
from flask import Flask
from flask_cors import CORS

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core.routes import blueprint
from deepface.commons.logger import Logger

# extra imports
import numpy as np
from flask.json.provider import DefaultJSONProvider


logger = Logger()

# custom JSON provider that handles numpy values
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
