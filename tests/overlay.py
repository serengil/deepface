import cv2
import matplotlib.pyplot as plt
from deepface.modules import streaming
from deepface import DeepFace

img_path = "dataset/img1.jpg"
img = cv2.imread(img_path)

overlay_img_path = "dataset/img6.jpg"
face_objs = DeepFace.extract_faces(overlay_img_path)
overlay_img = face_objs[0]["face"][:, :, ::-1] * 255

overlay_img = cv2.resize(overlay_img, (112, 112))

raw_img = img.copy()

demographies = DeepFace.analyze(img_path=img_path, actions=("age", "gender", "emotion"))
demography = demographies[0]

x = demography["region"]["x"]
y = demography["region"]["y"]
w = demography["region"]["w"]
h = demography["region"]["h"]

img = streaming.highlight_facial_areas(img=img, faces_coordinates=[(x, y, w, h)])

img = streaming.overlay_emotion(
    img=img,
    emotion_probas=demography["emotion"],
    x=x,
    y=y,
    w=w,
    h=h,
)

img = streaming.overlay_age_gender(
    img=img,
    apparent_age=demography["age"],
    gender=demography["dominant_gender"][0:1],
    x=x,
    y=y,
    w=w,
    h=h,
)

img = streaming.overlay_identified_face(
    img=img,
    target_img=overlay_img,
    label="angelina",
    x=x,
    y=y,
    w=w,
    h=h,
)

plt.imshow(img[:, :, ::-1])
plt.show()
