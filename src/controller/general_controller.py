from extension.utilServices import (send_json_response)
from extension.constants import CONSTANTS
import csv
import re
from flask_restplus import Namespace, Resource
import datetime
import copy

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
            'description': 'The app uses artificial intelligence (convolutional neural networks) that have been trained on dog/cat images to identify whether a dog/cat is in an image, and if so the species type (breed) and it\'s emotion.',
            'contact': 'https://eresearch.unimelb.edu.au',
            'developedByHTML': '<p>This app was developed by the Melbourne eResearch Group (<a href=\"www.eresearch.unimelb.edu.au\" target=\"_blank\">www.eresearch.unimelb.edu.au</a>) within the School of Computing and Information Systems (<a href=\"https://cis.unimelb.edu.au\" target=\"_blank\">https://cis.unimelb.edu.au</a>) at The University of Melbourne (<a href=\"www.unimelb.edu.au\" target=\"_blank\">www.unimelb.edu.au</a>).</p>',
            'descriptionHTML': '<p>The app uses artificial intelligence (convolutional neural networks) that have been trained on dog/cat images to identify whether a dog/cat is in an image, and if so the species type (breed) and it\'s emotion.</p>',
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

        total_cat_breed = []
        total_dog_breed = []
        prediction_collection = Prediction._get_collection()
        feedback_collection = Feedback._get_collection()
        statistical_data = copy.deepcopy(CONSTANTS['STATISTICAL_DATA'])

        # initial statistical_data based on breed and alpha order
        f = open("model_data/pet_classes.txt", "r")
        for each_line in f:
            first_char = each_line[0]
            if first_char.isupper():
                total_cat_breed.append(re.sub(r"_", " ", each_line.rstrip(), flags=re.IGNORECASE).title())
            else:
                total_dog_breed.append(re.sub(r"_", " ", each_line.rstrip(), flags=re.IGNORECASE).title())
        f.close()
        total_cat_breed.sort()
        total_dog_breed.sort()

        for each_breed in total_cat_breed:
            statistical_data['Cat']['prediction_data']['breed'][each_breed] = 0
            statistical_data['Cat']['feedback_data']['breed']['wrong'][each_breed] = 0
            statistical_data['Cat']['feedback_data']['breed']['correct'][each_breed] = 0

        for each_breed in total_dog_breed:
            statistical_data['Dog']['prediction_data']['breed'][each_breed] = 0
            statistical_data['Dog']['feedback_data']['breed']['wrong'][each_breed] = 0
            statistical_data['Dog']['feedback_data']['breed']['correct'][each_breed] = 0

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
                statistical_data[each_result['type']]['prediction_number'] = statistical_data[each_result['type']].get(
                    'prediction_number', 0) + 1
                statistical_data[each_result['type']]['prediction_data']['breed'][each_result['breed']] = \
                    statistical_data[each_result['type']]['prediction_data']['breed'].get(each_result['breed'], 0) + 1
                statistical_data[each_result['type']]['prediction_data']['emotion'][each_result['emotion']] = \
                    statistical_data[each_result['type']]['prediction_data']['emotion'].get(each_result['emotion'],
                                                                                            0) + 1
        # feedback
        total_feedback = feedback_collection.find({})

        for each_feedback in total_feedback:
            feedback_content = each_feedback['content']
            for each_content in feedback_content:
                statistical_data[each_content['type']]['feedback_number'] = statistical_data[each_content['type']].get(
                    'feedback_number', 0) + 1

                if not each_content['breedCorrectness']:
                    statistical_data[each_content['type']]['feedback_data']['breed']['wrong'][
                        each_content['breedFeedback']] = \
                        statistical_data[each_content['type']]['feedback_data']['breed'][
                            'wrong'].get(each_content['breedFeedback'], 0) + 1
                else:
                    statistical_data[each_content['type']]['feedback_data']['breed']['correct'][
                        each_content['breedFeedback']] = \
                        statistical_data[each_content['type']]['feedback_data']['breed']['correct'].get(
                            each_content['breedFeedback'], 0) + 1

                if not each_content['emotionCorrectness']:
                    statistical_data[each_content['type']]['feedback_data']['emotion']['wrong'][
                        each_content['emotionFeedback']] = \
                        statistical_data[each_content['type']]['feedback_data']['emotion']['wrong'].get(
                            each_content['emotionFeedback'], 0) + 1
                else:
                    statistical_data[each_content['type']]['feedback_data']['emotion']['correct'][
                        each_content['emotionFeedback']] = \
                        statistical_data[each_content['type']]['feedback_data']['emotion']['correct'].get(
                            each_content['emotionFeedback'], 0) + 1

        result = {
            'totalNumberOfPhotoUploaded': prediction_collection.count_documents({}),
            'totalNumberOfCatPrediction': statistical_data['Cat']['prediction_number'],
            'totalNumberOfDogPrediction': statistical_data['Dog']['prediction_number'],
            'totalNumberOfCatFeedback': statistical_data['Cat']['feedback_number'],
            'totalNumberOfDogFeedback': statistical_data['Dog']['feedback_number'],
            'totalNumberOfDogBreedPrediction': statistical_data['Dog']['prediction_data']['breed'],
            'totalNumberOfDogBreedCorrectFeedback': statistical_data['Dog']['feedback_data']['breed']['correct'],
            'totalNumberOfDogBreedWrongFeedback': statistical_data['Dog']['feedback_data']['breed']['wrong'],
            'totalNumberOfDogEmotionPrediction': statistical_data['Dog']['prediction_data']['emotion'],
            'totalNumberOfDogEmotionCorrectFeedback': statistical_data['Dog']['feedback_data']['emotion']['correct'],
            'totalNumberOfDogEmotionWrongFeedback': statistical_data['Dog']['feedback_data']['emotion']['wrong'],
            'totalNumberOfCatBreedPrediction': statistical_data['Cat']['prediction_data']['breed'],
            'totalNumberOfCatBreedCorrectFeedback': statistical_data['Cat']['feedback_data']['breed']['correct'],
            'totalNumberOfCatBreedWrongFeedback': statistical_data['Cat']['feedback_data']['breed']['wrong'],
            'totalNumberOfCatEmotionPrediction': statistical_data['Cat']['prediction_data']['emotion'],
            'totalNumberOfCatEmotionCorrectFeedback': statistical_data['Cat']['feedback_data']['emotion']['correct'],
            'totalNumberOfCatEmotionWrongFeedback': statistical_data['Cat']['feedback_data']['emotion']['wrong'],
            'numberOfPhotoByDate': total_number_of_photo_within_one_week
        }

        return send_json_response(result, 200)
