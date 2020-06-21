# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

**deepface** is a lightweight facial analysis framework including [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and demography ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) for Python. You can apply facial analysis with a few lines of code. It plans to bridge a gap between software engineering and machine learning studies.

## Installation

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/).

```python
pip install deepface
```

## Face Recognition 

A modern [face recognition pipeline](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: [detect](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). **DeepFace** handles all these common stages in the background.

**Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)

Verification function under the DeepFace interface offers a single face recognition. Each call of the function builds a face recognition model and this is very costly. If you are going to verify several faces sequentially, then you should pass an array of faces to the function instead of calling the function in a for loop. In this way, complex face recognition models will be built once and this will speed the function up dramatically. Besides, calling the function in a for loop might cause memory problems as well.

```python
from deepface import DeepFace
result  = DeepFace.verify("img1.jpg", "img2.jpg")
print("Is verified: ", result["verified"])

results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
print(results)
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

**Large scale face recognition** - [`Demo`](https://youtu.be/Hrjp-EStM_s)

You can apply face recognition on a [large scale](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) data set as well. Face recognition requires to apply face verification multiple times. Herein, deepface offers an out-of-the-box find function to handle this action. Representations of faces photos in your database folder will be stored in a pickle file when find function is called once. Then, deepface just finds representation of the target image. In this way, finding an identity in a large scale data set will be performed in just seconds.

```python
from deepface import DeepFace
import pandas as pd
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")
#dfs = DeepFace.find(img_path = ["img1.jpg", "img2.jpg"], db_path = "C:/workspace/my_db")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-6.jpg" width="95%" height="95%"></p>

**Supported face recognition models**

deepface currently wraps the **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/) and [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/). The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below. Accuracy and speed show difference based on the performing model.

```python
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]
for model in models:
   result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = model)
```

**Similarity**

Face recognition models are regular [convolutional neural networks](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/) and they are responsible to represent face photos as vectors. Decision of verification is based on the distance between vectors. We can classify pairs if its distance is less than a [threshold](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). 

Distance could be found by different metrics such as [Cosine Similarity](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
metrics = ["cosine", "euclidean", "euclidean_l2"]
for metric in metrics:
   result = DeepFace.verify("img1.jpg", "img2.jpg", distance_metric = metric)
```

**Ensemble learning for face recognition** - [`Demo`](https://youtu.be/EIBJJJ0ECXU)

A face recognition task can be handled by several models and similarity metrics. We can [combine](https://sefiks.com/2020/06/03/mastering-face-recognition-with-ensemble-learning/) the precictions of all of those models and metrics to improve the accuracy of a face recognition task. This offers a huge improvement on accuracy, precision and recall but it runs much slower than single models.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-4.jpg" width="95%" height="95%"></p>

```python
resp_obj = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Ensemble")
df = DeepFace.find(img_path = "img1.jpg", db_path = "my_db", model_name = "Ensemble")
```

**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise)and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Analysis function under the DeepFace interface is used to find demography of a face. 

```python
from deepface import DeepFace
demography = DeepFace.analyze("img4.jpg", actions = ['age', 'gender', 'race', 'emotion'])
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

**Streaming and Real Time Analysis** - [`Demo`](https://youtu.be/-c9sSJcx6wI)

You can run deepface for real time videos as well. 

Calling stream function under the DeepFace interface will access your webcam and apply both face recognition and facial attribute analysis. Stream function expects a database folder including face images. VGG-Face is the default face recognition model and cosine similarity is the default distance metric similar to verify function. The function starts to analyze if it can focus a face sequantially 5 frames. Then, it shows results 5 seconds.

```python
from deepface import DeepFace
DeepFace.stream("C:/User/Sefik/Desktop/database")
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

**API** - [`Demo`](https://youtu.be/HeKCQ6U9XmI)

Deepface serves an API as well. You can clone [`/api/api.py`](https://github.com/serengil/deepface/tree/master/api/api.py) and pass it to python command as an argument. This will get a rest service up. In this way, you can call deepface from an external system such as mobile app or web.

```
python api.py
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-api.jpg" width="90%" height="90%"></p>

The both face recognition and facial attribute analysis are covered in the API. You are expected to call these functions as http post methods. Service endpoints will be `http://127.0.0.1:5000/verify` for face recognition and `http://127.0.0.1:5000/analyze` for facial attribute analysis. You should pass input images as base64 encoded string in this case. [Here](https://github.com/serengil/deepface/tree/master/api), you can find a postman project.

**Passing pre-built face recognition models**

You can build models once and pass to deepface functions as well. This speeds you up if you are going to call deepface several times.

```python
#face recognition
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
model = VGGFace.loadModel() #all face recognition models have loadModel() function in their interfaces
DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", model = model)

#facial analysis
import json
from deepface.extendedmodels import Age, Gender, Race, Emotion
models = {}
models["emotion"] = Emotion.loadModel()
models["age"] = Age.loadModel()
models["gender"] = Gender.loadModel()
models["race"] = Race.loadModel()
DeepFace.analyze("img1.jpg", models=models)
```

## E-Learning

Deepface package for python is mentioned in this [playlist](https://www.youtube.com/watch?v=KRCvkNCOphE&list=PLsS_1RYmYQQFdWqxQggXHynP1rqaYXv_E) as video lectures. Subscribe the channel to stay up-to-date and be informed when a new lecture is added, 

## Disclaimer

Reference face recognition models have different type of licenses. This framework is just a wrapper for those models. That's why, licence types are inherited as well. You should check the licenses for the face recognition models before use.

Herein, [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE) is licensed under Apache License 2.0. [FB DeepFace](https://github.com/swghosh/DeepFace) and [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) is licensed under MIT License. The both Apache License 2.0 and MIT license types allow you to use for commercial purpose. 

On the other hand, [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is licensed under Creative Commons Attribution License. That's why, it is restricted to adopt VGG-Face for commercial use.

## Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

You can also support this project through Patreon.

<a href="https://www.patreon.com/bePatron?u=31795557"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png"></img></a>

## Citation

Please cite deepface in your publications if it helps your research. Here is an example BibTeX entry:

```
@misc{serengil2020deepface,
  abstract = {A Lightweight Face Recognition and Facial Attribute Analysis Framework for Python},
  author={Serengil, Sefik Ilkin},
  title={deepface},
  url = {https://github.com/serengil/deepface},
  year={2020}
}
```

## Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.

[Logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/). Licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
