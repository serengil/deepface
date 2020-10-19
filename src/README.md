# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)
[![License](http://img.shields.io/:license-MIT-blue.svg?style=flat)](https://github.com/serengil/deepface/blob/master/LICENSE)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

**deepface** is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). The library is mainly based on Keras and TensorFlow.

## Installation

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/).

```python
pip install deepface
```

## Face Recognition 

A modern [**face recognition pipeline**](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: [detect](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). **deepface** handles all these common stages in the background.

**Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)

Verification function under the deepface interface offers to verify face pairs as same person or different persons. You should pass face pairs as array instead of calling verify function in a for loop for the best practice. This will speed the function up dramatically and reduce the allocated memory.

```python
from deepface import DeepFace
result  = DeepFace.verify("img1.jpg", "img2.jpg")
#results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
print("Is verified: ", result["verified"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

**Large scale face recognition** - [`Demo`](https://youtu.be/Hrjp-EStM_s) 

Face recognition requires to apply face verification several times. Herein, deepface offers an out-of-the-box find function to handle this action. You can apply face recognition on a [large scale](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) data set as well.

```python
from deepface import DeepFace
import pandas as pd
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")
#dfs = DeepFace.find(img_path = ["img1.jpg", "img2.jpg"], db_path = "C:/workspace/my_db")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-6-v2.jpg" width="95%" height="95%"></p>

**Face recognition models** - [`Demo`](https://youtu.be/i_MOwvhbLdI)

Deepface is a **hybrid** face recognition package. It currently wraps the **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). The default configuration verifies faces with **VGG-Face** model. You can set the base model while verification as illustared below.

```python
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib"]
for model in models:
   result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = model)
```

FaceNet, VGG-Face and Dlib [overperforms](https://youtu.be/i_MOwvhbLdI) than OpenFace, DeepFace and DeepID based on experiments.

**Similarity**

Face recognition models are regular [convolutional neural networks](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/) and they are responsible to represent faces as vectors. Decision of verification is based on the distance between vectors. We can classify pairs if its distance is less than a [threshold](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). 

Distance could be found by different metrics such as [Cosine Similarity](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the **cosine similarity**. You can alternatively set the similarity metric while verification as demostratred below.

```python
metrics = ["cosine", "euclidean", "euclidean_l2"]
for metric in metrics:
   result = DeepFace.verify("img1.jpg", "img2.jpg", distance_metric = metric)
```

Euclidean L2 form [seems](https://youtu.be/i_MOwvhbLdI) to be more stable than cosine and regular Euclidean distance based on experiments.

**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Analysis function under the DeepFace interface is used to find demography of a face. 

```python
from deepface import DeepFace
obj = DeepFace.analyze("img4.jpg", actions = ['age', 'gender', 'race', 'emotion'])
#objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

**Face Detectors** - [`Demo`](https://youtu.be/GZ2p2hj2H5k)

Face detection and alignment are early stages of a modern face recognition pipeline. [OpenCV haar cascade](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [SSD](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [Dlib](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) and [MTCNN](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) methods are wrapped in deepface as a detector. You can optionally pass a custom detector to functions in deepface interface. OpenCV is the default detector if you won't pass a detector.

```python
backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
for backend in backends:
   #face detection and alignment
   detected_face = DeepFace.detectFace("img.jpg", detector_backend = backend)
   
   #face verification
   obj = DeepFace.verify("img1.jpg", "img2.jpg", detector_backend = backend)
   
   #face recognition
   df = DeepFace.find(img_path = "img.jpg", db_path = "my_db", detector_backend = backend)
   
   #facial analysis
   demography = DeepFace.analyze("img4.jpg", detector_backend = backend)
```

[MTCNN](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) seems to overperform in detection and alignment stages but it is slower than [SSD](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/).

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

**Ensemble learning for face recognition** - [`Demo`](https://youtu.be/EIBJJJ0ECXU)

A face recognition task can be handled by several models and similarity metrics. Herein, deepface offers a [special boosting and combination solution](https://sefiks.com/2020/06/03/mastering-face-recognition-with-ensemble-learning/) to improve the accuracy of a face recognition task. This provides a huge improvement on accuracy metrics. Human beings could have 97.53% score for face recognition tasks whereas this ensemble method passes the human level accuracy and gets 98.57% accuracy. On the other hand, this runs much slower than single models.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-4.jpg" width="70%" height="70%"></p>

```python
resp_obj = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Ensemble")
df = DeepFace.find(img_path = "img1.jpg", db_path = "my_db", model_name = "Ensemble")
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

Deepface package for python is mentioned in this [playlist](https://www.youtube.com/watch?v=KRCvkNCOphE&list=PLsS_1RYmYQQFdWqxQggXHynP1rqaYXv_E) as video lectures. **Subscribe** the channel to stay up-to-date and be informed when a new lecture is added.

## Translations

You can also read a translated version of deepface tutorials in [Chinese (深臉)](https://zhuanlan.zhihu.com/p/151403935) or [Turkish](https://bilisim.io/2020/03/26/deepface-python-icin-yuz-tanima-ve-demografi-analizi-iskeleti/).

## Disclaimer

Reference face recognition models have different type of licenses. This framework is just a wrapper for those models. That's why, licence types are inherited as well. You should check the licenses for the face recognition models before use.

Herein, [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE) is licensed under Apache License 2.0. [FB DeepFace](https://github.com/swghosh/DeepFace) and [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) is licensed under MIT License. [Dlib](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt) is licensed under Boost Software License. They all allow you to use for personal and commercial purpose free. 

On the other hand, [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) is licensed under Creative Commons Attribution License. That's why, it is restricted to adopt VGG-Face for commercial use.

## Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

## Citation

Please cite deepface in your publications if it helps your research. Here is an example BibTeX entry:

```
@inproceedings{serengil2020lightface,
  title={LightFace: A Hybrid Deep Face Recognition Framework},
  author={Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  year={2020},
  organization={IEEE}
}
```

## Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details.

[Logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/). Licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
