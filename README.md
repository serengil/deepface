# deepface

**deepface** is a lightweight python based face recognition framework. You can verify faces with just a few lines of codes.

```python
from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")
```

# Face recognition models

Face recognition can be handled by different models. Currently, [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Facenet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/) and [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/) models are supported in deepface. The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below. Accuracy and speed show difference based on the performing model.

```python
vggface_result = DeepFace.verify("img1.jpg", "img2.jpg")
#vggface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face")
facenet_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Facenet")
openface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "OpenFace")
```

# Similarity

These models actually find the vector embeddings of faces. Decision of verification is based on the distance between vectors. Distance could be found by different metrics such as [`Cosine Similarity`](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "cosine")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean_l2")
```

# Verification

Verification function returns a tuple including boolean verification result, distance between two faces and max threshold to identify. 

```
(True, 0.281734, 0.30)
```

You can just check the verification result to decide that two images are same person or not. Thresholds for distance metrics are already tuned in the framework for face recognition models and distance metrics.

```python
verified = result[0] #returns True if images are same person's face
```

Instead of using pre-tuned threshold values, you can alternatively check the distance by yourself.

```python
distance = result[1] #the less the better
threshold = 0.30 #threshold for VGG-Face and Cosine Similarity
if distance < threshold:
	return True
else:
	return False
```

# Installation

The easiest way to install deepface is to download it from [PyPI](https://pypi.org/project/deepface/).

```
pip install deepface
```

Alternatively, you can directly download the source code from this repository. GitHub repo might be newer than the PyPI version.

```
git clone https://github.com/serengil/chefboost.git
cd chefboost
pip install -e .
```

Initial tests are run for Python 3.5.5 on Windows 10 but this is an OS-independent framework. Even though pip handles to install dependent libraries, the framework basically needs the following dependencies. You might need the following library requirements if you install the source code from github.

```
pip install numpy==1.14.0
pip install matplotlib==2.2.2
pip install gdown==3.10.1
pip install opencv-python==3.4.4
pip install tensorflow==1.9.0
pip install keras==2.2.0
```

# Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

# Licence

Chefboost is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.
