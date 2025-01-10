# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
from tqdm import tqdm

# project dependencies
from deepface.modules import modeling, detection, preprocessing
from deepface.models.demography import Gender, Race, Emotion

# pylint: disable=trailing-whitespace
def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2' (default is cosine).

        align (boolean): Perform alignment based on the eye positions (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
           the analysis results for a detected face.

           Each dictionary in the list contains the following keys:

           - 'region' (dict): Represents the rectangular region of the detected face in the image.
               - 'x': x-coordinate of the top-left corner of the face.
               - 'y': y-coordinate of the top-left corner of the face.
               - 'w': Width of the detected face region.
               - 'h': Height of the detected face region.

           - 'age' (float): Estimated age of the detected face.

           - 'face_confidence' (float): Confidence score for the detected face.
                Indicates the reliability of the face detection.

           - 'dominant_gender' (str): The dominant gender in the detected face.
                Either "Man" or "Woman."

           - 'gender' (dict): Confidence scores for each gender category.
               - 'Man': Confidence score for the male gender.
               - 'Woman': Confidence score for the female gender.

           - 'dominant_emotion' (str): The dominant emotion in the detected face.
                Possible values include "sad," "angry," "surprise," "fear," "happy,"
                "disgust," and "neutral."

           - 'emotion' (dict): Confidence scores for each emotion category.
               - 'sad': Confidence score for sadness.
               - 'angry': Confidence score for anger.
               - 'surprise': Confidence score for surprise.
               - 'fear': Confidence score for fear.
               - 'happy': Confidence score for happiness.
               - 'disgust': Confidence score for disgust.
               - 'neutral': Confidence score for neutrality.

           - 'dominant_race' (str): The dominant race in the detected face.
                Possible values include "indian," "asian," "latino hispanic,"
                "black," "middle eastern," and "white."

           - 'race' (dict): Confidence scores for each race category.
               - 'indian': Confidence score for Indian ethnicity.
               - 'asian': Confidence score for Asian ethnicity.
               - 'latino hispanic': Confidence score for Latino/Hispanic ethnicity.
               - 'black': Confidence score for Black ethnicity.
               - 'middle eastern': Confidence score for Middle Eastern ethnicity.
               - 'white': Confidence score for White ethnicity.
    """

    # if actions is passed as tuple with single item, interestingly it becomes str here
    if isinstance(actions, str):
        actions = (actions,)

    # check if actions is not an iterable or empty.
    if not hasattr(actions, "__getitem__") or not actions:
        raise ValueError("`actions` must be a list of strings.")

    actions = list(actions)

    # For each action, check if it is valid
    for action in actions:
        if action not in ("emotion", "age", "gender", "race"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )
    # ---------------------------------
    resp_objects = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    # Anti-spoofing check
    if anti_spoofing and any(img_obj.get("is_real", True) is False for img_obj in img_objs):
        raise ValueError("Spoof detected in the given image.")

    # Prepare the input for the model
    valid_faces = []
    face_regions = []
    face_confidences = []
    
    for img_obj in img_objs:
        # Extract the face content
        img_content = img_obj["face"]
        # Check if the face content is empty
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            continue

        # Convert the image to RGB format from BGR
        img_content = img_content[:, :, ::-1]
        # Resize the image to the target size for the model
        img_content = preprocessing.resize_image(img=img_content, target_size=(224, 224))

        valid_faces.append(img_content)
        face_regions.append(img_obj["facial_area"])
        face_confidences.append(img_obj["confidence"])

    # If no valid faces are found, return an empty list
    if not valid_faces:
        return []

    # Convert the list of valid faces to a numpy array
    faces_array = np.array(valid_faces)

    # Create placeholder response objects for each face
    for _ in range(len(valid_faces)):
        resp_objects.append({})


    # For each action, predict the corresponding attribute
    pbar = tqdm(
        range(0, len(actions)),
        desc="Finding actions",
        disable=silent if len(actions) > 1 else True,
    )
    
    for index in pbar:
        action = actions[index]
        pbar.set_description(f"Action: {action}")

        if action == "emotion":
            # Build the emotion model
            model = modeling.build_model(task="facial_attribute", model_name="Emotion")
            emotion_predictions = model.predict(faces_array)
            
            for idx, predictions in enumerate(emotion_predictions):
                sum_of_predictions = predictions.sum()
                resp_objects[idx]["emotion"] = {}
                
                for i, emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = 100 * predictions[i] / sum_of_predictions
                    resp_objects[idx]["emotion"][emotion_label] = emotion_prediction
                
                resp_objects[idx]["dominant_emotion"] = Emotion.labels[np.argmax(predictions)]

        elif action == "age":
            # Build the age model
            model = modeling.build_model(task="facial_attribute", model_name="Age")
            age_predictions = model.predict(faces_array)
            
            for idx, age in enumerate(age_predictions):
                resp_objects[idx]["age"] = int(age)

        elif action == "gender":
            # Build the gender model
            model = modeling.build_model(task="facial_attribute", model_name="Gender")
            gender_predictions = model.predict(faces_array)
            
            for idx, predictions in enumerate(gender_predictions):
                resp_objects[idx]["gender"] = {}
                
                for i, gender_label in enumerate(Gender.labels):
                    gender_prediction = 100 * predictions[i]
                    resp_objects[idx]["gender"][gender_label] = gender_prediction
                
                resp_objects[idx]["dominant_gender"] = Gender.labels[np.argmax(predictions)]

        elif action == "race":
            # Build the race model
            model = modeling.build_model(task="facial_attribute", model_name="Race")
            race_predictions = model.predict(faces_array)
            
            for idx, predictions in enumerate(race_predictions):
                sum_of_predictions = predictions.sum()
                resp_objects[idx]["race"] = {}
                
                for i, race_label in enumerate(Race.labels):
                    race_prediction = 100 * predictions[i] / sum_of_predictions
                    resp_objects[idx]["race"][race_label] = race_prediction
                
                resp_objects[idx]["dominant_race"] = Race.labels[np.argmax(predictions)]

    # Add the face region and confidence to the response objects
    for idx, resp_obj in enumerate(resp_objects):
        resp_obj["region"] = face_regions[idx]
        resp_obj["face_confidence"] = face_confidences[idx]

    return resp_objects
