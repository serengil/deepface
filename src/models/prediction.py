from config import db


class Prediction(db.Document):
    meta = {'collection': 'predictions'}
    predictionResults = db.ListField(required=True)
    rawPredictionResults = db.ListField(required=True)
    date = db.DateTimeField(required=True)
