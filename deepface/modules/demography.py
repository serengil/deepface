# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
from tqdm import tqdm

# project dependencies
from deepface.modules import modeling
from deepface.commons import functions
from deepface.extendedmodels import Gender, Race, Emotion


def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    silent: bool = False,
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
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        align (boolean): Perform alignment based on the eye positions.

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
        the analysis results for a detected face. Example:

        [
                {
                        "region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
                        "age": 28.66,
                        'face_confidence': 0.9993908405303955,
                        "dominant_gender": "Woman",
                        "gender": {
                                'Woman': 99.99407529830933,
                                'Man': 0.005928758764639497,
                        }
                        "dominant_emotion": "neutral",
                        "emotion": {
                                'sad': 37.65260875225067,
                                'angry': 0.15512987738475204,
                                'surprise': 0.0022171278033056296,
                                'fear': 1.2489334680140018,
                                'happy': 4.609785228967667,
                                'disgust': 9.698561953541684e-07,
                                'neutral': 56.33133053779602
                        }
                        "dominant_race": "white",
                        "race": {
                                'indian': 0.5480832420289516,
                                'asian': 0.7830780930817127,
                                'latino hispanic': 2.0677512511610985,
                                'black': 0.06337375962175429,
                                'middle eastern': 3.088453598320484,
                                'white': 93.44925880432129
                        }
                }
        ]
    """
    # ---------------------------------
    # validate actions
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

    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img_content, img_region, img_confidence in img_objs:
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
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
                    emotion_predictions = modeling.build_model("Emotion").predict(img_content)
                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {}
                    for i, emotion_label in enumerate(Emotion.labels):
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        obj["emotion"][emotion_label] = emotion_prediction

                    obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

                elif action == "age":
                    apparent_age = modeling.build_model("Age").predict(img_content)
                    # int cast is for exception - object of type 'float32' is not JSON serializable
                    obj["age"] = int(apparent_age)

                elif action == "gender":
                    gender_predictions = modeling.build_model("Gender").predict(img_content)
                    obj["gender"] = {}
                    for i, gender_label in enumerate(Gender.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj["gender"][gender_label] = gender_prediction

                    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

                elif action == "race":
                    race_predictions = modeling.build_model("Race").predict(img_content)
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
