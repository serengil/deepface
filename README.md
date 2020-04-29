# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

**Deepface** is a lightweight facial analysis framework including [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and demography ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) for Python. You can apply facial analysis with a few lines of code. It plans to bridge a gap between software engineering and machine learning studies.

# Installation

The easiest way to install deepface is to download it from [PyPI](https://pypi.org/project/deepface/).

```python
pip install deepface
```

# Face Recognition

Verify function under the DeepFace interface is used for face recognition.

```python
from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")

print("Is verified: ", result["verified"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

Modern face recognition pipelines consist of 4 stages: detect, [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), represent and verify. Deepface handles all these common stages in the background.

Each call of verification function builds a face recognition model from scratch and this is a costly operation. If you are going to verify multiple faces sequentially, then you should pass an array of faces to verification function to speed the operation up. In this way, complex face recognition models will be built once.

```python
dataset = [
	['dataset/img1.jpg', 'dataset/img2.jpg'],
	['dataset/img1.jpg', 'dataset/img3.jpg']
]
result = DeepFace.verify(dataset)
```

## Face recognition models

Face recognition can be handled by different models. Currently, [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/) and [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/) models are supported in deepface. The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below. Accuracy and speed show difference based on the performing model.

```python
vggface_result = DeepFace.verify("img1.jpg", "img2.jpg") #default is VGG-Face
#vggface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face") #identical to the line above
facenet_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Facenet")
openface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "OpenFace")
deepface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "DeepFace")
```

The complexity and response time of each face recognition model is different so do accuracy scores. Mean ± std. dev. of 7 runs on CPU for each model in my experiments is illustrated in the following table.

| Model        | VGG-Face         | OpenFace         | Google FaceNet   | Facebook DeepFace |
| ---          | ---              | ---              | ---              | ---               |
| Building     | 2.35 s ± 46.9 ms | 6.37 s ± 1.28 s  | 25.7 s ± 7.93 s  | 23.9 s ± 2.52 s   |
| Verification | 897 ms ± 38.3 ms | 616 ms ± 12.1 ms | 684 ms ± 7.69 ms | 605 ms ± 13.2 ms  |

## Passing pre-built models

You can build a face recognition model once and pass this to verify function as well. This might be logical if you need to call verify function several times.

```python
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
model = VGGFace.loadModel() #all face recognition models have loadModel() function in their interfaces
DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", model = model)
```

## Similarity

These models actually find the vector embeddings of faces. In other words, we use face recognition models as [autoencoders](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/). Decision of verification is based on the distance between vectors. Distance could be found by different metrics such as [`Cosine Similarity`](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "cosine")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean_l2")
```

# Facial Attribute Analysis

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`emotion`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise)and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Analysis function under the DeepFace interface is used to find demography of a face.

```python
from deepface import DeepFace
demography = DeepFace.analyze("img4.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time

print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

Model building and prediction times are different for those facial analysis models. Mean ± std. dev. of 7 runs on CPU for each model in my experiments is illustrated in the following table.

| Model      | Emotion          | Age              | Gender           | Race             |
| ---        | ---              | ---              | ---              | ---              |
| Building   | 243 ms ± 15.2 ms | 2.25 s ± 34.9    | 2.25 s ± 90.9 ms | 2.23 s ± 68.6 ms |
| Prediction | 389 ms ± 11.4 ms | 524 ms ± 16.1 ms | 516 ms ± 10.8 ms | 493 ms ± 20.3 ms |

## Passing pre-built models

You can build facial attribute analysis models once and pass these to analyze function as well. This might be logical if you need to call analyze function several times.

```python
import json
from deepface.extendedmodels import Age, Gender, Race, Emotion

models = {}
models["emotion"] = Emotion.loadModel()
models["age"] = Age.loadModel()
models["gender"] = Gender.loadModel()
models["race"] = Race.loadModel()

DeepFace.analyze("img1.jpg", models=models)
```

# Streaming and Real Time Analysis

You can run deepface for real time videos as well. Calling stream function under the DeepFace interface will access your webcam and apply both face recognition and facial attribute analysis. Stream function expects a database folder including face images. VGG-Face is the default face recognition model and cosine similarity is the default distance metric similar to verify function. The function starts to analyze if it can focus a face sequantially 5 frames. Then, it shows results 5 seconds.

```python
from deepface import DeepFace
DeepFace.stream("/user/database")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-3.jpg" width="90%" height="90%"></p>

Even though face recognition is based on one-shot learning, you can use multiple face pictures of a person as well. You should rearrange your directory structure as illustrated below.

```bash
user
├── database
│   ├── Alice
│   │   ├── Alice1.jpg
│   │   ├── Alice2.jpg
│   ├── Bob
│   │   ├── Bob.jpg
```

BTW, you should use regular slash ( / ) instead of backslash ( \ ) in Windows OS while passing the path to stream function. E.g. `DeepFace.stream("C:/User/Sefik/Desktop/database")`.

# API

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-api.jpg" width="90%" height="90%"></p>

Deepface [serves](https://youtu.be/HeKCQ6U9XmI) an API as well. You can clone [`/api/api.py`](https://github.com/serengil/deepface/tree/master/api/api.py) and pass it to python command as an argument. This will get a rest service up.

```
python api.py
```

The both face recognition and facial attribute analysis are covered in the API. You are expected to call these functions as http post methods. Service endpoints will be `http://127.0.0.1:5000/verify` for face recognition and `http://127.0.0.1:5000/analyze` for facial attribute analysis. You should pass input images as base64 encoded string in this case. [Here](https://github.com/serengil/deepface/tree/master/api), you can find a postman project.

# Playlist

Deepface is mentioned in this [youtube playlist](https://www.youtube.com/watch?v=KRCvkNCOphE&list=PLsS_1RYmYQQFdWqxQggXHynP1rqaYXv_E).

# Disclaimer

Reference face recognition models have different type of licenses. This framework is just a wrapper for those models. That's why, licence types are inherited as well. You should check the licenses for the face recognition models before use.

Herein, [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE) is licensed under Apache License 2.0. [FB DeepFace](https://github.com/swghosh/DeepFace) and [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) is licensed under MIT License. The both Apache License 2.0 and MIT license types allow you to use for commercial purpose. 

On the other hand, [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is licensed under Creative Commons Attribution License. That's why, it is restricted to adopt VGG-Face for commercial use.

# Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

You can also support this project through Patreon.

<a href="https://www.patreon.com/bePatron?u=31795557"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png"></img></a>

# Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.

[Logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/). Licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
