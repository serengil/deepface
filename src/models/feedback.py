from config import db


class FeedbackContent(db.EmbeddedDocument):
    
    type = db.StringField(required=True)
    ageFeedback = db.StringField(required=True)
    ageCorrectness = db.BooleanField(required=True)
    genderFeedback = db.StringField(required=True)
    genderCorrectness = db.BooleanField(required=True)
    emotionFeedback = db.StringField(required=True)
    emotionCorrectness = db.BooleanField(required=True)


class Feedback(db.Document):
    meta = {'collection': 'feedbacks'}
    content = db.EmbeddedDocumentListField('FeedbackContent', required=True)
    date = db.DateTimeField(required=True)
