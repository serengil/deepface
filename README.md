# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)

**deepface** is a lightweight python based facial analysis framework including face recognition and demography (age, gender, emotion and race). You can use the framework with a just few lines of codes.

# Face Recognition

Verify function under the DeepFace interface is used for face recognition.

```python
from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")
```

## Face recognition models

Face recognition can be handled by different models. Currently, [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Facenet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/) and [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/) models are supported in deepface. The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below. Accuracy and speed show difference based on the performing model.

```python
vggface_result = DeepFace.verify("img1.jpg", "img2.jpg") #default is VGG-Face
#vggface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face")
facenet_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Facenet")
openface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "OpenFace")
```

## Similarity

These models actually find the vector embeddings of faces. Decision of verification is based on the distance between vectors. Distance could be found by different metrics such as [`Cosine Similarity`](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "cosine")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean_l2")
```

VGG-Face has the highest accuracy score but it is not convenient for real time studies because of its complex structure. Facenet is a complex model as well. On the other hand, OpenFace has a close accuracy score but it performs the fastest. That's why, OpenFace is much more convenient for real time studies.

## Verification

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

# Facial Attribute Analysis

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`emotion`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) predictions. Analysis function under the DeepFace interface is used to find demography of a face.

```python
from deepface import DeepFace
demography = DeepFace.analyze("img.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img.jpg", ['age', 'gender', 'race', 'emotion']) #identical to above line
```

Analysis function returns a json object.

```
{
   "age": 31.940666721338523
   , "gender": "Woman"
   , "race": {
      "asian": 11.314528435468674,
      "indian": 17.498773336410522,
      "black": 3.541698679327965,
      "white": 21.96589708328247,
      "middle eastern": 19.87851709127426,
      "latino hispanic": 25.800585746765137
   }
   , "dominant_race": "latino hispanic"
   , "emotion": {
      "angry": 6.004959843039945e-16,
      "disgust": 4.9082449499136944e-34,
      "fear": 4.7907148065142067e-23,
      "happy": 100.0,
      "sad": 4.8685008000541987e-14,
      "surprise": 5.66862615875019e-10,
      "neutral": 3.754812086254056e-09
   }
   , "dominant_emotion": "happy"
}
```

Then, you can retrieve the fields of the response object easily in Python.

```python
import json
print("Age: ",demography["age"])
```

# Installation

The easiest way to install deepface is to download it from [PyPI](https://pypi.org/project/deepface/).

```
pip install deepface
```

Alternatively, you can directly download the source code from this repository. **GitHub repo might be newer than the PyPI version**.

```
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

Initial tests are run for Python 3.5.5 on Windows 10 but this is an OS-independent framework. Even though pip handles to install dependent libraries, the framework basically needs the following dependencies. You might need the following library requirements if you install the source code from github.

```
pip install numpy==1.14.0
pip install pandas==0.23.4
pip install matplotlib==2.2.2
pip install gdown==3.10.1
pip install opencv-python==3.4.4
pip install tensorflow==1.9.0
pip install keras==2.2.0
pip install tqdm==4.30.0
```

# Disclaimer

Reference face recognition models have different type of licenses. This framework is just a wrapper for those models. That's why, licence types are inherited as well. You should check the licenses for the face recognition models before use.

Herein, [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE) is licensed under Apache License 2.0, and [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) is licensed under MIT License. They both allow you to use commercial use. On the other hand, [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is licensed under Creative Commons Attribution License. That's why, it is restricted to adopt VGG-Face for commercial use.

# Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

# Licence

Chefboost is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.
