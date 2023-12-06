import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger()

# ----------------------------------------------
# build face recognition model

model_name = "VGG-Face"

model = DeepFace.build_model(model_name=model_name)

target_size = functions.find_target_size(model_name)

logger.info(f"target_size: {target_size}")

# ----------------------------------------------
# load images and find embeddings

img1 = DeepFace.extract_faces(img_path="dataset/img1.jpg", target_size=target_size)[0]["face"]
img1 = np.expand_dims(img1, axis=0)  # to (1, 224, 224, 3)
img1_representation = model.predict(img1)[0, :]

img2 = DeepFace.extract_faces(img_path="dataset/img3.jpg", target_size=target_size)[0]["face"]
img2 = np.expand_dims(img2, axis=0)
img2_representation = model.predict(img2)[0, :]

# ----------------------------------------------
# distance between two images

distance_vector = np.square(img1_representation - img2_representation)
logger.debug(distance_vector)

distance = np.sqrt(distance_vector.sum())
logger.info(f"Euclidean distance: {distance}")

# ----------------------------------------------
# expand vectors to be shown better in graph

img1_graph = []
img2_graph = []
distance_graph = []

for i in range(0, 200):
    img1_graph.append(img1_representation)
    img2_graph.append(img2_representation)
    distance_graph.append(distance_vector)

img1_graph = np.array(img1_graph)
img2_graph = np.array(img2_graph)
distance_graph = np.array(distance_graph)

# ----------------------------------------------
# plotting

fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
plt.imshow(img1[0])
plt.axis("off")

ax2 = fig.add_subplot(3, 2, 2)
im = plt.imshow(img1_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

ax3 = fig.add_subplot(3, 2, 3)
plt.imshow(img2[0])
plt.axis("off")

ax4 = fig.add_subplot(3, 2, 4)
im = plt.imshow(img2_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

ax5 = fig.add_subplot(3, 2, 5)
plt.text(0.35, 0, f"Distance: {distance}")
plt.axis("off")

ax6 = fig.add_subplot(3, 2, 6)
im = plt.imshow(distance_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

plt.show()

# ----------------------------------------------
