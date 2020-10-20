from flask import request
import base64
from PIL import Image
import io
import datetime
from extension.utilServices import send_json_response
from flask_restplus import Namespace, Resource


api = Namespace('predict', path='/predict', description='prediction related operations')


@api.route('')
class Prediction(Resource):
    @api.doc('make prediction')
    def post(self):
        """
        return prediction results and save it to the database
        :return: prediction results
        """
        from models.prediction import Prediction
        from config import deepface
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded)).convert('RGB')

        img, detections = deepface.analyze(image)

        # encode image and jsonify detections
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        base64_string = img_str.decode('utf-8')

        result = {
            'img_str': base64_string,
            'results': detections,
            'message': '',
            'status': 'success'
        }

        if len(detections) == 0:
            result['message'] = "We’re not very sure of what this may be, could you try with another image",
            result['status'] = 'failure'
        elif len(detections) == 1:
            result['isShowId'] = 'false'

        if len(detections) > 0:
            formatted_prediction_results = []
            for each in detections:
                age = each['age']
                gender = each['gender']
                emotion = each['emotion']['dominant']
                emotion_score = each['emotion']['dominant_score']
                formatted_prediction_results.append({
                    'age': age,
                    'gender': gender,
                    'emotion': emotion,
                    'emotionScore': emotion_score
                })

            # store results to db
            new_prediction = Prediction(**{
                'predictionResults': formatted_prediction_results,
                'rawPredictionResults': detections,
                'date': datetime.datetime.now(),
            })
            new_prediction.save()

        return send_json_response(result, 200)
