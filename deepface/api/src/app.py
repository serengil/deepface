# 3rd party dependencies
from flask import Flask
from flask_cors import CORS
import numpy as np
from flask.json.provider import DefaultJSONProvider

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core.routes import blueprint
from deepface.commons.logger import Logger

logger = Logger()


# This is the custom JSON provider that will fix the serialization error
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        # np.generic is the base class for all numpy scalar types
        if isinstance(o, np.generic):
            return o.item()
        # Handle numpy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
        # Let the base class default method raise the TypeError for other types
        return super(NumpyJSONProvider, self).default(o)


def create_app():
    app = Flask(__name__)
    
    # Set our custom JSON provider to handle numpy types
    app.json = NumpyJSONProvider(app)
    
    CORS(app)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
