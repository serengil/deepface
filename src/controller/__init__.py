from flask_restplus import Api
from flask import Blueprint
from controller.prediction_controller import api as predict_apis
from controller.feedback_controller import api as feedback_apis
from controller.general_controller import api as general_apis


# old api, will be discard later
api_bp_old = Blueprint('api_old', __name__)

api = Api(api_bp_old)

api.add_namespace(predict_apis)
api.add_namespace(general_apis)

"""
new api with /api/ path
"""
api_bp_new = Blueprint('api_new', __name__)

api_new = Api(api_bp_new)
api_new.add_namespace(general_apis)
api_new.add_namespace(predict_apis)
api_new.add_namespace(feedback_apis)
