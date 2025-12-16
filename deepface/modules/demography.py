# built-in dependencies
from typing import List, Optional, Union, IO

# 3rd party dependencies
import cv2
import numpy as np
from tqdm import tqdm

# project dependencies
from deepface.modules import modeling, detection
from deepface.models.Face import Face, FaceDemographics
from deepface.models.demography import Age, Gender, Race, Emotion


def analyze(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    actions: list = ["emotion", "age", "gender", "race"],
    detector_backend: str = "opencv",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
    silent: bool = False,
) -> List[Face]:
    """
    Analyze facial attributes such as age, gender, emotion, and race in the provided image.
    Args:
        img_path (str, np.ndarray, IO[bytes], list): The exact path to the image, a numpy array
            in BGR format, a file object that supports at least `.read` and is opened in binary
            mode, or a base64 encoded image. If the source image contains multiple faces,
            the result will include information for each detected face.

        actions (tuple): Attributes to analyze. The default is ('age', 'gender', 'emotion', 'race').
            You can exclude some of these attributes from the analysis if needed.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n',  'yolov11s', 'yolov11m',
            'centerface' (default is opencv).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

        max_faces (int): Set a limit on the number of faces to be processed and returned (default is None).

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

    Returns:
        List[Face.Face]: A list of Face objects, each containing demographic results for detected faces.
    """

    # batch input
    if (isinstance(img_path, np.ndarray) and img_path.ndim == 4 and img_path.shape[0] > 1) or isinstance(img_path, list):
        batch_faces: List[Face] = []
        # Execute analysis for each image in the batch.
        for single_img in img_path:
            # Call the analyze function for each image in the batch.
            faces = analyze(
                img_path=single_img,
                actions=actions,
                detector_backend=detector_backend,
                anti_spoofing=anti_spoofing,
                max_faces=max_faces,
                silent=silent,
            )
            # Append the response to the batch response list.
            batch_faces.extend(faces)
        return batch_faces

    # for each action, check if it is valid
    for action in actions:
        if action not in ("emotion", "age", "gender", "race"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )

    # extract faces
    img, faces = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )

    # load requested models
    emotion_model: Optional[Emotion.EmotionClient] = None
    age_model: Optional[Age.ApparentAgeClient] = None
    gender_model: Optional[Gender.GenderClient] = None
    race_model: Optional[Race.RaceClient] = None
    if "emotion" in actions:
        emotion_model = modeling.build_model(task="facial_attribute", model_name="Emotion")
    if "age" in actions:
        age_model = modeling.build_model(task="facial_attribute", model_name="Age")
    if "gender" in actions:
        gender_model = modeling.build_model(task="facial_attribute", model_name="Gender")
    if "race" in actions:
        race_model = modeling.build_model(task="facial_attribute", model_name="Race")

    # analyze each face
    for face in faces:
        # init empty demographics
        face.demographics = FaceDemographics(
            emotions={},
            genders={},
            races={},
            age=None,
        )

        # prepare input image for facial attribute analysis
        # 224x224 is the expected input for all demography models except emotion (which uses 48x48 grayscale internally)
        face_img = face.get_face_crop(img, target_size=(224,224), scale_face_area=1, eyes_aligned=True)
        face_img = face_img[:, :, ::-1] # RGB to BGR (Note: we are not sure why we do that unforunately)

        # run requested facial attribute analysis
        pbar = tqdm(
            range(0, len(actions)),
            desc="Finding actions",
            disable=silent if len(actions) > 1 else True,
        )
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Action: {action}")

            if action == "emotion" and emotion_model is not None:
                # Note: emotion predict resizes the image to (48, 48) and converts to grayscale internally
                emotion_predictions = emotion_model.predict(face_img)
                sum_of_predictions = emotion_predictions.sum()
                for i, emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = emotion_predictions[i] / sum_of_predictions
                    face.demographics.emotions[emotion_label] = emotion_prediction

            elif action == "age" and age_model is not None:
                apparent_age = age_model.predict(face_img)
                face.demographics.age = int(apparent_age)

            elif action == "gender" and gender_model is not None:
                gender_predictions = gender_model.predict(face_img)
                for i, gender_label in enumerate(Gender.labels):
                    gender_prediction = gender_predictions[i]
                    face.demographics.genders[gender_label] = gender_prediction

            elif action == "race" and race_model is not None:
                race_predictions = race_model.predict(face_img)
                sum_of_predictions = race_predictions.sum()
                for i, race_label in enumerate(Race.labels):
                    race_prediction = race_predictions[i] / sum_of_predictions
                    face.demographics.races[race_label] = race_prediction

    return faces
