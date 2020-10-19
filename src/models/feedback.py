from config import db


class FeedbackContent(db.EmbeddedDocument):
    type = db.StringField(required=True)
    breedFeedback = db.StringField(required=True)
    breedCorrectness = db.BooleanField(required=True)
    emotionFeedback = db.StringField(required=True)
    emotionCorrectness = db.BooleanField(required=True)


class Feedback(db.Document):
    meta = {'collection': 'feedbacks'}
    content = db.EmbeddedDocumentListField('FeedbackContent', required=True)
    date = db.DateTimeField(required=True)
