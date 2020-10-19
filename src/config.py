from flask import Flask
from flask_mongoengine import MongoEngine
from deepface.DeepFaceLite import DeepFaceLite
# import tensorflow as tf
# from tensorflow.python.keras.backend import set_session
import os

"""
initialize the ML engine
"""
db = MongoEngine()
# sess = tf.Session()
# graph = tf.get_default_graph()
# set_session(sess)
deepface = DeepFaceLite()

"""
database environment variable
"""
DB_NAME = os.environ.get('HAP_DB_NAME')
DB_HOST = os.environ.get('HAP_DB_HOST')
DB_PORT = os.environ.get('HAP_DB_PORT')
DB_USERNAME = os.environ.get('HAP_DB_USERNAME')
DB_PASSWORD = os.environ.get('HAP_DB_PASSWORD')


def create_app():
    """
    return app instance
    :return: app instance
    """
    app = Flask(__name__)
    if DB_USERNAME is not None and DB_PASSWORD is not None:
        app.config['MONGODB_SETTINGS'] = {
            'db': DB_NAME,
            'host': DB_HOST,
            'port': int(DB_PORT),
            'username': DB_USERNAME,
            'password': DB_PASSWORD
        }
    else:
        app.config['MONGODB_SETTINGS'] = {
            'db': DB_NAME,
            'host': DB_HOST,
            'port': int(DB_PORT)
        }

    db.init_app(app)

    return app
