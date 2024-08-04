# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
from tqdm import tqdm

# project dependencies
from deepface.modules import modeling, detection, preprocessing
from deepface.extendedmodels import Gender, Race, Emotion


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
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
            (default is opencv).

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

    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")

        img_content = img_obj["face"]
        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            continue

        # rgb to bgr
        img_content = img_content[:, :, ::-1]

        # resize input image
        img_content = preprocessing.resize_image(img=img_content, target_size=(224, 224))

        obj = {}
        # facial attribute analysis
        pbar = tqdm(
            range(0, len(actions)),
            desc="Finding actions",
            disable=silent if len(actions) > 1 else True,
        )
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Action: {action}")

            if action == "emotion":
                emotion_predictions = modeling.build_model(
                    task="facial_attribute", model_name="Emotion"
                ).predict(img_content)
                sum_of_predictions = emotion_predictions.sum()

                obj["emotion"] = {}
                for i, emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                    obj["emotion"][emotion_label] = emotion_prediction

                obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

            elif action == "age":
                apparent_age = modeling.build_model(
                    task="facial_attribute", model_name="Age"
                ).predict(img_content)
                # int cast is for exception - object of type 'float32' is not JSON serializable
                obj["age"] = int(apparent_age)

            elif action == "gender":
                gender_predictions = modeling.build_model(
                    task="facial_attribute", model_name="Gender"
                ).predict(img_content)
                obj["gender"] = {}
                for i, gender_label in enumerate(Gender.labels):
                    gender_prediction = 100 * gender_predictions[i]
                    obj["gender"][gender_label] = gender_prediction

                obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

            elif action == "race":
                race_predictions = modeling.build_model(
                    task="facial_attribute", model_name="Race"
                ).predict(img_content)
                sum_of_predictions = race_predictions.sum()

                obj["race"] = {}
                for i, race_label in enumerate(Race.labels):
                    race_prediction = 100 * race_predictions[i] / sum_of_predictions
                    obj["race"][race_label] = race_prediction

                obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]

            # -----------------------------
            # mention facial areas
            obj["region"] = img_region
            # include image confidence
            obj["face_confidence"] = img_confidence

        resp_objects.append(obj)

    return resp_objects
