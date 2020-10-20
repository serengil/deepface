from extension.utilServices import (send_json_response)
from extension.constants import CONSTANTS
import csv
import re
from flask_restplus import Namespace, Resource
import datetime
import copy
import math

api = Namespace('general', path='/', description='general information related to app')


@api.route('/get-app-info')
class AppInfo(Resource):
    @api.doc('return the app developer information')
    def get(self):
        """
        return the footer information
        :return: return the app related information
        """
        app_info = {
            'developedBy': 'This app was developed by the Melbourne eResearch Group (www.eresearch.unimelb.edu.au) within the School of Computing and Information Systems (https://cis.unimelb.edu.au) at The University of Melbourne (www.unimelb.edu.au). ',
            'description': 'The app uses artificial intelligence (convolutional neural networks) to identify the age, gender, and emotion of the people.',
            'contact': 'https://eresearch.unimelb.edu.au',
            'developedByHTML': '<p>This app was developed by the Melbourne eResearch Group (<a href=\"www.eresearch.unimelb.edu.au\" target=\"_blank\">www.eresearch.unimelb.edu.au</a>) within the School of Computing and Information Systems (<a href=\"https://cis.unimelb.edu.au\" target=\"_blank\">https://cis.unimelb.edu.au</a>) at The University of Melbourne (<a href=\"www.unimelb.edu.au\" target=\"_blank\">www.unimelb.edu.au</a>).</p>',
            'descriptionHTML': '<p>The app uses artificial intelligence (convolutional neural networks) to identify the age, gender, and emotion of the people.</p>',
            'contactHTML': '<p>Please contact us at: <a href=\"eresearch.unimelb.edu.au\" target=\"_blank\">eresearch.unimelb.edu.au</a></p>'
        }

        return send_json_response(app_info, 200)


@api.route('/get-statistical-results')
class StatisticalData(Resource):
    @api.doc('return statistical data')
    def get(self):
        """
        return the statistical information
        query across feedback and prediction model
        :return: return statistical information
        """
        from models.prediction import Prediction
        from models.feedback import Feedback

        prediction_collection = Prediction._get_collection()
        feedback_collection = Feedback._get_collection()
        statistical_data = copy.deepcopy(CONSTANTS['STATISTICAL_DATA'])

        # initial statistical_data based on age and ascending order
        total_age = []
        for i in range(21):
            total_age.append(str(i * 5) + '+')

        for each_age in total_age:
            statistical_data['prediction_data']['age'][each_age] = 0
            statistical_data['feedback_data']['age']['wrong'][each_age] = 0
            statistical_data['feedback_data']['age']['correct'][each_age] = 0

        # photo by date
        today = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        total_number_of_photo_within_one_week = {
            (today - datetime.timedelta(days=6)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today - datetime.timedelta(days=5), '$gte': today - datetime.timedelta(days=6)}}),
            (today - datetime.timedelta(days=5)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today - datetime.timedelta(days=4), '$gte': today - datetime.timedelta(days=5)}}),
            (today - datetime.timedelta(days=4)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today - datetime.timedelta(days=3), '$gte': today - datetime.timedelta(days=4)}}),
            (today - datetime.timedelta(days=3)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today - datetime.timedelta(days=2), '$gte': today - datetime.timedelta(days=3)}}),
            (today - datetime.timedelta(days=2)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today - datetime.timedelta(days=1), '$gte': today - datetime.timedelta(days=2)}}),
            (today - datetime.timedelta(days=1)).strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$lt': today, '$gte': today - datetime.timedelta(days=1)}}),
            today.strftime('%d/%m/%Y'): prediction_collection.count_documents(
                {'date': {'$gte': today}}),
        }

        # prediction
        total_prediction = prediction_collection.find({})

        for each_prediction in total_prediction:
            prediction_results = each_prediction['predictionResults']
            for each_result in prediction_results:
                statistical_data['prediction_number'] = statistical_data.get(
                    'prediction_number', 0) + 1

                # age, gender, emotion
                each_age = str(5*math.floor(each_result['age'] / 5)) + '+'

                statistical_data['prediction_data']['age'][each_age] = \
                    statistical_data['prediction_data']['age'].get(each_age, 0) + 1
                statistical_data['prediction_data']['gender'][each_result['gender']] = \
                    statistical_data['prediction_data']['gender'].get(each_result['gender'], 0) + 1
                statistical_data['prediction_data']['emotion'][each_result['emotion']] = \
                    statistical_data['prediction_data']['emotion'].get(each_result['emotion'], 0) + 1

        # feedback
        total_feedback = feedback_collection.find({})

        for each_feedback in total_feedback:
            feedback_content = each_feedback['content']
            for each_content in feedback_content:
                statistical_data['feedback_number'] = statistical_data.get(
                    'feedback_number', 0) + 1

                # age
                if each_content['ageCorrectness'] == False:
                    statistical_data['feedback_data']['age']['wrong'][
                        each_content['ageFeedback']] = \
                        statistical_data['feedback_data']['age'][
                            'wrong'].get(each_content['ageFeedback'], 0) + 1
                else:
                    statistical_data['feedback_data']['age']['correct'][
                        each_content['ageFeedback']] = \
                        statistical_data['feedback_data']['age']['correct'].get(
                            each_content['ageFeedback'], 0) + 1

                # gender
                if each_content['genderCorrectness'] == False:
                    statistical_data['feedback_data']['gender']['wrong'][
                        each_content['genderFeedback']] = \
                        statistical_data['feedback_data']['gender'][
                            'wrong'].get(each_content['genderFeedback'], 0) + 1
                else:
                    statistical_data['feedback_data']['gender']['correct'][
                        each_content['genderFeedback']] = \
                        statistical_data['feedback_data']['gender']['correct'].get(
                            each_content['genderFeedback'], 0) + 1

                if each_content['emotionCorrectness'] == False:
                    statistical_data['feedback_data']['emotion']['wrong'][
                        each_content['emotionFeedback']] = \
                        statistical_data['feedback_data']['emotion']['wrong'].get(
                            each_content['emotionFeedback'], 0) + 1
                else:
                    statistical_data['feedback_data']['emotion']['correct'][
                        each_content['emotionFeedback']] = \
                        statistical_data['feedback_data']['emotion']['correct'].get(
                            each_content['emotionFeedback'], 0) + 1

        result = {
            'totalNumberOfPhotoUploaded': prediction_collection.count_documents({}),

            'totalNumberOfPrediction': statistical_data['prediction_number'],
            'totalNumberOfFeedback': statistical_data['feedback_number'],

            'totalNumberOfAgePrediction': statistical_data['prediction_data']['age'],
            'totalNumberOfAgeCorrectFeedback': statistical_data['feedback_data']['age']['correct'],
            'totalNumberOfAgeWrongFeedback': statistical_data['feedback_data']['age']['wrong'],

            'totalNumberOfGenderPrediction': statistical_data['prediction_data']['gender'],
            'totalNumberOfGenderCorrectFeedback': statistical_data['feedback_data']['gender']['correct'],
            'totalNumberOfGenderWrongFeedback': statistical_data['feedback_data']['gender']['wrong'],

            'totalNumberOfEmotionPrediction': statistical_data['prediction_data']['emotion'],
            'totalNumberOfEmotionCorrectFeedback': statistical_data['feedback_data']['emotion']['correct'],
            'totalNumberOfEmotionWrongFeedback': statistical_data['feedback_data']['emotion']['wrong'],

            'numberOfPhotoByDate': total_number_of_photo_within_one_week
        }

        return send_json_response(result, 200)
