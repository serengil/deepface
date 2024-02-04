# 3rd parth dependencies
from flask import Flask

try:
    # unit tests
    from api.src.modules.core.routes import blueprint
except:
    # runtime
    from modules.core.routes import blueprint


def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    return app
