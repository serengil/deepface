# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon.png" width="200" height="200"></p>

**Deepface** is a lightweight facial analysis framework including face recognition and demography (age, gender, emotion and race) for Python. Modern face recognition pipelines consist of 4 stages: detect, [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), represent and verify. Deepface handles all these common stages in the background. You can use the framework with a just few lines of codes

# Face Recognition

Verify function under the DeepFace interface is used for face recognition.

```python
from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/test-case-2.jpg"  width="50%" height="50%"></p>

```json
{
   "verified": true,
   "distance": 0.25638097524642944,
   "max_threshold_to_verify": 0.40,
   "model": "VGG-Face",
   "similarity_metric": "cosine"
}
```

You can retrieve the fields of the response object easily in Python.

```python
print("Is verified: ", result["verified"])
print("Distance: ", result["distance"])
```

Each call of verification function builds a face recognition model scratch and this is a costly operation. If you are going to verify multiple faces sequentially, then you should pass an array to verify function to speed up.

```python
dataset = [
	['dataset/img1.jpg', 'dataset/img2.jpg'],
	['dataset/img5.jpg', 'dataset/img6.jpg']
]
result = DeepFace.verify(dataset)
```

## Face recognition models

Face recognition can be handled by different models. Currently, [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google Facenet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/) and [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/) models are supported in deepface. The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below. Accuracy and speed show difference based on the performing model.

```python
vggface_result = DeepFace.verify("img1.jpg", "img2.jpg") #default is VGG-Face
#vggface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face") #identical to the line above
facenet_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Facenet")
openface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "OpenFace")
deepface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "DeepFace")
```

VGG-Face has the highest accuracy score but it is not convenient for real time studies because of its complex structure. FaceNet is a complex model as well. On the other hand, OpenFace has a close accuracy score but it performs the fastest. That's why, OpenFace is much more convenient for real time studies.

## Similarity

These models actually find the vector embeddings of faces. Decision of verification is based on the distance between vectors. Distance could be found by different metrics such as [`Cosine Similarity`](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "cosine")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean_l2")
```

# Facial Attribute Analysis

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`emotion`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) predictions. Analysis function under the DeepFace interface is used to find demography of a face.

```python
from deepface import DeepFace
demography = DeepFace.analyze("img4.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above

demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/test-case-1.jpg" width="20%" height="20%"></p>

Analysis function returns a json object.

```json
{
"age": 32.49221594557578,
"gender": "Woman",
"race": {
   "asian": 3.928472101688385, 
   "white": 55.44567108154297, 
   "middle eastern": 15.896821022033691, 
   "indian": 3.050043433904648, 
   "latino hispanic": 20.90577930212021, 
   "black": 0.7732132915407419
},
"dominant_race": "white",
"emotion": {
   "angry": 3.1055836006999016, 
   "fear": 1.1844050139188766, 
   "neutral": 86.2661361694336, 
   "sad": 7.137920707464218, 
   "disgust": 0.0001227657776325941, 
   "happy": 2.245445176959038, 
   "surprise": 0.06038688006810844
}, 
"dominant_emotion": "neutral"
}
```

Then, you can retrieve the fields of the response object easily in Python.

```python
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
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
pip install Pillow==5.2.0
```

# Playlist

Deepface is mentioned in this [youtube playlist](https://www.youtube.com/watch?v=KRCvkNCOphE&list=PLsS_1RYmYQQFdWqxQggXHynP1rqaYXv_E).

# Disclaimer

Reference face recognition models have different type of licenses. This framework is just a wrapper for those models. That's why, licence types are inherited as well. You should check the licenses for the face recognition models before use.

Herein, [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE) is licensed under Apache License 2.0. [FB DeepFace](https://github.com/swghosh/DeepFace) and [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) is licensed under MIT License. The both Apache License 2.0 and MIT license types allow you to use for commercial purpose. 

On the other hand, [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is licensed under Creative Commons Attribution License. That's why, it is restricted to adopt VGG-Face for commercial use.

# Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

# Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.

[Logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/). Licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
