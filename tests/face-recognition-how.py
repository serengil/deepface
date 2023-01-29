import matplotlib.pyplot as plt
import numpy as np
from deepface.basemodels import VGGFace
from deepface.commons import functions

# ----------------------------------------------
# build face recognition model

model = VGGFace.loadModel()

try:
    input_shape = model.layers[0].input_shape[1:3]
except:  # issue 470
    input_shape = model.layers[0].input_shape[0][1:3]

print("model input shape: ", model.layers[0].input_shape[1:])
print("model output shape: ", model.layers[-1].input_shape[-1])

# ----------------------------------------------
# load images and find embeddings

# img1 = functions.detectFace("dataset/img1.jpg", input_shape)
img1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
img1_representation = model.predict(img1)[0, :]

# img2 = functions.detectFace("dataset/img3.jpg", input_shape)
img2 = functions.preprocess_face("dataset/img3.jpg", input_shape)
img2_representation = model.predict(img2)[0, :]

# ----------------------------------------------
# distance between two images

distance_vector = np.square(img1_representation - img2_representation)
# print(distance_vector)

distance = np.sqrt(distance_vector.sum())
print("Euclidean distance: ", distance)

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
plt.imshow(img1[0][:, :, ::-1])
plt.axis("off")

ax2 = fig.add_subplot(3, 2, 2)
im = plt.imshow(img1_graph, interpolation="nearest", cmap=plt.cm.ocean)
plt.colorbar()

ax3 = fig.add_subplot(3, 2, 3)
plt.imshow(img2[0][:, :, ::-1])
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
