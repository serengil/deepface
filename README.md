# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)
[![Stars](https://img.shields.io/github/stars/serengil/deepface)](https://github.com/serengil/deepface)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/deepface/blob/master/LICENSE)
[![Patreon](https://img.shields.io/:support-patreon-orange.svg?style=flat)](https://www.patreon.com/bePatron?u=31795557&redirect_uri=https%3A%2F%2Fgithub.com%2Fserengil%2Fdeepface)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

Deepface is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). The library is mainly based on Keras and TensorFlow.

## Installation

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/).

```python
pip install deepface
```

## Face Recognition 

A modern [**face recognition pipeline**](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: [detect](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). Deepface handles all these common stages in the background. You can just call its verification, find or analysis function in its interface with a single line of code.

[Here](https://youtu.be/WnUVYQP4h44), you can find an introduction video covering its functionalities and the best practices.

**Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)

Verification function under the deepface interface offers to verify face pairs as same person or different persons. You should pass face pairs as array instead of calling verify function in a for loop for the best practice. This will speed the function up dramatically and reduce the allocated memory.

```python
from deepface import DeepFace
result  = DeepFace.verify("img1.jpg", "img2.jpg")
#results = DeepFace.verify([['img1.jpg', 'img2.jpg'], ['img1.jpg', 'img3.jpg']])
print("Is verified: ", result["verified"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

Herein, face pairs could be exact image paths, numpy array or base64 encoded images.

**Face recognition** - [`Demo`](https://youtu.be/Hrjp-EStM_s) 

Face recognition requires to apply face verification several times. Herein, deepface offers an out-of-the-box find function to handle this action. It stores the representations of your facial database and you don't have to find it again and again. In this way, you can apply [face recognition](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) data set as well. The find function returns pandas data frame if a single image path is passed, and it returns list of pandas data frames if list of image paths are passed.

```python
from deepface import DeepFace
import pandas as pd
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")
#dfs = DeepFace.find(img_path = ["img1.jpg", "img2.jpg"], db_path = "C:/workspace/my_db")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-6-v2.jpg" width="95%" height="95%"></p>

Herein, image path argument could be exact image path, numpy array or base64 encoded image. Also, you are expected to store your facial image data base in the folder that you passed to the db_path argument with .jpg or .png extension. 

**Face recognition models** - [`Demo`](https://youtu.be/i_MOwvhbLdI)

Deepface is a **hybrid** face recognition package. It currently wraps the **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). The default configuration verifies faces with VGG-Face model. You can set the base model while verification as illustared below.

```python
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
for model in models:
   result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = model)
   df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db", model_name = model)
```

FaceNet, VGG-Face, ArcFace and Dlib [overperforms](https://youtu.be/i_MOwvhbLdI) than OpenFace, DeepFace and DeepID based on experiments. Supportively, FaceNet got 99.65%; ArcFace got 99.40%; Dlib got 99.38%; VGG-Face got 98.78%; OpenFace got 93.80% accuracy scores on [LFW data set](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/) whereas human beings could have just 97.53%.

**Similarity**

Face recognition models are regular [convolutional neural networks](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/) and they are responsible to represent faces as vectors. Decision of verification is based on the distance between vectors. We can classify pairs if its distance is less than a [threshold](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). 

Distance could be found by different metrics such as [Cosine Similarity](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration finds the cosine similarity. You can alternatively set the similarity metric while verification as demostratred below.

```python
metrics = ["cosine", "euclidean", "euclidean_l2"]
for metric in metrics:
   result = DeepFace.verify("img1.jpg", "img2.jpg", distance_metric = metric)
   df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db", distance_metric = metric)
```

Euclidean L2 form [seems](https://youtu.be/i_MOwvhbLdI) to be more stable than cosine and regular Euclidean distance based on experiments.

**Tech Stack** - [`Vlog`](https://youtu.be/R8fHsL7u3eE)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/tech-stack.png" width="90%" height="90%"></p>

Recommended tech stack for face verification is mainly based on [relational databases and regular SQL](https://sefiks.com/2021/02/06/deep-face-recognition-with-sql/) or key-value stores such as [Redis](https://sefiks.com/2021/03/02/deep-face-recognition-with-redis/) or [Cassandra](https://sefiks.com/2021/01/24/deep-face-recognition-with-cassandra/). Herein, key-value stores overperform than regular relational databases.

Face verification is a subset of face recognition. In other words, you can run any face verification tool for face recognition as well. However, face verification has O(1) and face recognition has O(n) time complexity. That's why, face recognition becomes problematic with regular face verification tools on millions/billions level data and limited hardware.

You should use some big data solutions in face recognition when the data becomes huge. NoSQL databases comes with the power of map reduce technology. Here, [Hadoop](https://sefiks.com/2021/01/31/deep-face-recognition-with-hadoop-and-spark/) with Spark or Hive will overperform if you have lots of datanodes and clusters. Besides, [mongoDb](https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/) is a document database and it is highly scalable as well.

On the other hand, approximate nearest neighbor (a-nn) algorithm reduces the time complexity dramatically. [Spotify Annoy](https://sefiks.com/2020/09/16/large-scale-face-recognition-with-spotify-annoy/), [Facebook Faiss](https://sefiks.com/2020/09/17/large-scale-face-recognition-with-facebook-faiss/) and [NMSLIB](https://sefiks.com/2020/09/19/large-scale-face-recognition-with-nmslib/) are amazing a-nn libraries. Besides, [Elasticsearch](https://sefiks.com/2020/11/27/large-scale-face-recognition-with-elasticsearch/) wraps an a-nn algorithm and it offers highly scalability feature. You should run deepface within those a-nn frameworks if you have really large scale data sets. Those libraries come with high speed but they don't guarantee to find the closest ones always in contrast to k-nn algorithm run in nosql databases.

Finally, graph databases offer to discover relations hard to find. [Neo4j](https://sefiks.com/2021/04/03/deep-face-recognition-with-neo4j/) is a pretty graph database exploring indirect relations between facial images.

Here, you can find some implementation demos of deepface with a-nn libraries: [`Elasticsearch`](https://youtu.be/i4GvuOmzKzo) and [`Spotify Annoy`](https://youtu.be/Jpxm914o2xk); key-value stores: [`Redis`](https://youtu.be/eo-fTv4eYzo), [`Cassandra`](https://youtu.be/VQqHs6-4Ylg); and graph databases: [`Neo4j`](https://youtu.be/X-hB2kBFBXs).

**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)

Deepface also offers facial attribute analysis including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Analysis function under the DeepFace interface is used to find demography of a face. 

```python
from deepface import DeepFace
obj = DeepFace.analyze(img_path = "img4.jpg", actions = ['age', 'gender', 'race', 'emotion'])
#objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

Age model got ± 4.65 MAE; gender model got 97.44% accuracy, 96.29% precision and 95.05% recall as mentioned in its [tutorial](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/).

Herein, image path argument could be exact image path, numpy array or base64 encoded image.

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

**Face Detectors** - [`Demo`](https://youtu.be/GZ2p2hj2H5k)

Face detection and alignment are early stages of a modern face recognition pipeline. [`OpenCV`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [`SSD`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) and [`MTCNN`](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) methods are wrapped in deepface as a detector. You can optionally pass a custom detector to functions in deepface interface. MTCNN is the default detector if you won't pass any detector.

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

**Passing pre-built face recognition models**

You can build models once and pass to deepface functions as well. This speeds you up if you are going to call deepface several times. Consider this approach if you are going to call deepface functions in a for loop.

```python
#face recognition
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
for model_name in models:
   model = DeepFace.build_model(model_name)
   DeepFace.verify("img1.jpg", "img2.jpg", model_name = model_name, model = model)

#facial analysis
models = {}
actions = ['Age', 'Gender', 'Emotion', 'Race']
for action in actions:
   models[action.lower()] = DeepFace.build_model(action)
DeepFace.analyze("img1.jpg", models=models)
```

## FAQ and Troubleshooting

Pre-trained weights of custom models will be downloaded from Google Drive source to your environment once. Download limit of my Google Drive account might be exceeded sometimes. In this case, you might have an exception like "Too many users have viewed or downloaded this file recently. Please try accessing the file again later". You can still download the pre-trained weights from Google Drive manually. You should then download the pre-trained weights to {HOME_FOLDER}/.deepface/weights folder. It won't try to download the weight file if it exists in the weights folder. You can find out your HOME_FOLDER as shown below.

```python
from pathlib import Path
home = str(Path.home())
print("HOME_FOLDER is ",home)
```

## Contribution

Pull requests are welcome. You should run the unit tests locally by running [`test/unit_tests.py`](https://github.com/serengil/deepface/blob/master/tests/unit_tests.py). Please share the unit test result logs in the PR. Deepface is currently compatible with TF 1 and 2 versions. Change requests should satisfy those requirements both.

## Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one.

You can also support this project on [Patreon](https://www.patreon.com/bePatron?u=31795557&redirect_uri=https%3A%2F%2Fgithub.com%2Fserengil%2Fdeepface) 🙏

<p align="left"><a href="https://www.patreon.com/bePatron?u=31795557&redirect_uri=https%3A%2F%2Fgithub.com%2Fserengil%2Fdeepface"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/patreon.png" width="30%"></a></p>

## Citation

Please cite [deepface](https://ieeexplore.ieee.org/document/9259802) in your publications if it helps your research. Here is an example BibTeX entry:

```BibTeX
@inproceedings{serengil2020lightface,
  title={LightFace: A Hybrid Deep Face Recognition Framework},
  author={Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={23-27},
  year={2020},
  doi={10.1109/ASYU50717.2020.9259802},
  organization={IEEE}
}
```

## Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details. However, the library wraps some face recognition models: [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/), [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE), [DeepFace](https://github.com/swghosh/DeepFace), [ArcFace](https://github.com/leondgarse/Keras_insightface/blob/master/LICENSE) and [Dlib](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt). Licence types will be inherited if you are going to use those models.

Deepface [logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/) and it is licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
