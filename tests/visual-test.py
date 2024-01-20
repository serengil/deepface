import matplotlib.pyplot as plt
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
]
detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]


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

# extract faces
for detector_backend in detector_backends:
    face_objs = DeepFace.extract_faces(
        img_path="dataset/img1.jpg", detector_backend=detector_backend
    )
    for face_obj in face_objs:
        face = face_obj["face"]
        logger.info(detector_backend)
        plt.imshow(face)
        plt.axis("off")
        plt.show()
        logger.info("-----------")
