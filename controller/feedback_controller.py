from flask import request
from extension.utilServices import send_json_response
from flask_restplus import Namespace, Resource
import datetime
api = Namespace('feedback', path='/feedback', description='prediction related operations')


@api.route('')
class Feedback(Resource):
    @api.doc('feedback related api')
    def post(self):
        """
        store user feedbacks to the database
        :return: operation results
        """
        from models.feedback import Feedback, FeedbackContent

        message = request.get_json(force=True)

        try:
            new_feedback = Feedback()
            new_feedback_content = []
            for each_content in message['data']:
                feedback_content = FeedbackContent(**each_content)
                new_feedback_content.append(feedback_content)

            new_feedback.content = new_feedback_content
            new_feedback.date = datetime.datetime.now()
            new_feedback.save()
            return send_json_response({}, 200)
        except Exception as error:
            return send_json_response({'msg': str(error)}, 500)
