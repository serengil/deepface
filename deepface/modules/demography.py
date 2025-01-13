# built-in dependencies
from typing import Any, Dict, List, Union, Optional

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

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    if anti_spoofing and any(img_obj.get("is_real", True) is False for img_obj in img_objs):
        raise ValueError("Spoof detected in the given image.")

    def preprocess_face(img_obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Preprocess the face image for analysis.
        """
        img_content = img_obj["face"]
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            return None
        img_content = img_content[:, :, ::-1]  # BGR to RGB
        return preprocessing.resize_image(img=img_content, target_size=(224, 224))

    # Filter out empty faces
    face_data = [(preprocess_face(img_obj), img_obj["facial_area"], img_obj["confidence"]) 
                 for img_obj in img_objs if img_obj["face"].size > 0]
    
    if not face_data:
        return []

    # Unpack the face data
    valid_faces, face_regions, face_confidences = zip(*face_data)
    faces_array = np.array(valid_faces)

    # Initialize the results list with face regions and confidence scores
    results = [{"region": region, "face_confidence": conf} 
              for region, conf in zip(face_regions, face_confidences)]

    # Iterate over the actions and perform analysis
    pbar = tqdm(
        actions,
        desc="Finding actions",
        disable=silent if len(actions) > 1 else True,
    )

    for action in pbar:
        pbar.set_description(f"Action: {action}")
        model = modeling.build_model(task="facial_attribute", model_name=action.capitalize())
        predictions = model.predict(faces_array)

        # If the model returns a single prediction, reshape it to match the number of faces.
        # Determine the correct shape of predictions by using number of faces and predictions shape.
        # Example: For 1 face with Emotion model, predictions will be reshaped to (1, 7).
        if faces_array.shape[0] == 1 and len(predictions.shape) == 1:
            # For models like `Emotion`, which return a single prediction for a single face
            predictions = predictions.reshape(1, -1)

        # Update the results with the predictions
        # ----------------------------------------
        # For emotion, calculate the percentage of each emotion and find the dominant emotion
        if action == "emotion":
            emotion_results = [
                {
                    "emotion": {
                        label: 100 * pred[i] / pred.sum()
                        for i, label in enumerate(Emotion.labels)
                    },
                    "dominant_emotion": Emotion.labels[np.argmax(pred)]
                }
                for pred in predictions
            ]
            for result, emotion_result in zip(results, emotion_results):
                result.update(emotion_result)
        # ----------------------------------------
        # For age, find the dominant age category (0-100)
        elif action == "age":
            age_results = [{"age": int(np.argmax(pred) if len(pred.shape) > 0 else pred)} 
                         for pred in predictions]
            for result, age_result in zip(results, age_results):
                result.update(age_result)
        # ----------------------------------------
        # For gender, calculate the percentage of each gender and find the dominant gender
        elif action == "gender":
            gender_results = [
                {
                    "gender": {
                        label: 100 * pred[i]
                        for i, label in enumerate(Gender.labels)
                    },
                    "dominant_gender": Gender.labels[np.argmax(pred)]
                }
                for pred in predictions
            ]
            for result, gender_result in zip(results, gender_results):
                result.update(gender_result)
        # ----------------------------------------
        # For race, calculate the percentage of each race and find the dominant race
        elif action == "race":
            race_results = [
                {
                    "race": {
                        label: 100 * pred[i] / pred.sum()
                        for i, label in enumerate(Race.labels)
                    },
                    "dominant_race": Race.labels[np.argmax(pred)]
                }
                for pred in predictions
            ]
            for result, race_result in zip(results, race_results):
                result.update(race_result)

    return results
