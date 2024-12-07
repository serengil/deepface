# 3rd party dependencies
import matplotlib.pyplot as plt

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

# some models (e.g. Dlib) and detectors (e.g. retinaface) do not have test cases
# because they require to install huge packages
# this module is for local runs

model_names = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "Dlib",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
    "yolov8",
    "yolov11n",
    "yolov11s",
    "yolov11m"
]

detector_backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    # "mediapipe", # crashed in mac
    "retinaface",
    "yunet",
    "yolov8",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "centerface",
]

# verification
for model_name in model_names:
    obj = DeepFace.verify(
        img1_path="dataset/img1.jpg", img2_path="dataset/img2.jpg", model_name=model_name
    )
    logger.info(obj)
    logger.info("---------------------")

# represent
for model_name in model_names:
    embedding_objs = DeepFace.represent(img_path="dataset/img1.jpg", model_name=model_name)
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        logger.info(f"{model_name} produced {len(embedding)}D vector")


# find
dfs = DeepFace.find(
    img_path="dataset/img1.jpg", db_path="dataset", model_name="Facenet", detector_backend="mtcnn"
)
for df in dfs:
    logger.info(df)

expand_areas = [0]
img_paths = ["dataset/img11.jpg", "dataset/img11_reflection.jpg"]
for expand_area in expand_areas:
    for img_path in img_paths:
        # extract faces
        for detector_backend in detector_backends:
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                align=True,
                expand_percentage=expand_area,
            )
            for face_obj in face_objs:
                face = face_obj["face"]
                logger.info(f"testing {img_path} with {detector_backend}")
                logger.info(face_obj["facial_area"])
                logger.info(face_obj["confidence"])

                # we know opencv sometimes cannot find eyes
                if face_obj["facial_area"]["left_eye"] is not None:
                    assert isinstance(face_obj["facial_area"]["left_eye"], tuple)
                    assert isinstance(face_obj["facial_area"]["left_eye"][0], int)
                    assert isinstance(face_obj["facial_area"]["left_eye"][1], int)

                if face_obj["facial_area"]["right_eye"] is not None:
                    assert isinstance(face_obj["facial_area"]["right_eye"], tuple)
                    assert isinstance(face_obj["facial_area"]["right_eye"][0], int)
                    assert isinstance(face_obj["facial_area"]["right_eye"][1], int)

                # left eye is really the left eye of the person
                if (
                    face_obj["facial_area"]["left_eye"] is not None
                    and face_obj["facial_area"]["right_eye"] is not None
                ):
                    re_x = face_obj["facial_area"]["right_eye"][0]
                    le_x = face_obj["facial_area"]["left_eye"][0]
                    assert re_x < le_x, "right eye must be the right eye of the person"

                type_conf = type(face_obj["confidence"])
                assert isinstance(
                    face_obj["confidence"], float
                ), f"confidence type must be float but it is {type_conf}"
                assert face_obj["confidence"] <= 1

                plt.imshow(face)
                plt.axis("off")
                plt.show()
                logger.info("-----------")
