# deepface

[![Downloads](https://pepy.tech/badge/deepface)](https://pepy.tech/project/deepface)
[![Stars](https://img.shields.io/github/stars/serengil/deepface)](https://github.com/serengil/deepface)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/deepface/blob/master/LICENSE)

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

Deepface is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). Those models already reached and passed the human level accuracy. The library is mainly based on Keras and TensorFlow.

## Installation

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/). It's going to install the library itself and its prerequisites as well.

```python
pip install deepface
```

Then you will be able to import the library and use its functionalities.

```python
from deepface import DeepFace
```

**Facial Recognition** - [`Demo`](https://youtu.be/WnUVYQP4h44)

A modern [**face recognition pipeline**](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 4 common stages: [detect](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). Deepface handles all these common stages in the background. You can just call its verification, find or analysis function with a single line of code.

**Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)

This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or based64 encoded images is also welcome.

```python
result  = DeepFace.verify("img1.jpg", "img2.jpg")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

**Face recognition** - [`Demo`](https://youtu.be/Hrjp-EStM_s)

[Face recognition](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) requires applying face verification many times. Herein, deepface has an out-of-the-box find function to handle this action. It's going to look for the identity of input image in the database path and it will return pandas data frame as output.

```python
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-6-v2.jpg" width="95%" height="95%"></p>

**Face recognition models** - [`Demo`](https://youtu.be/i_MOwvhbLdI)

Deepface is a **hybrid** face recognition package. It currently wraps many **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/) and [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/). The default configuration uses VGG-Face model.

```python
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = models[1])
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db", model_name = models[1])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-wrapped-models.png" width="95%" height="95%"></p>

FaceNet, VGG-Face, ArcFace and Dlib [overperforms](https://youtu.be/i_MOwvhbLdI) than OpenFace, DeepFace and DeepID based on experiments. Supportively, FaceNet (/w 512d) got 99.65%; ArcFace got 99.40%; Dlib got 99.38%; VGG-Face got 98.78%; OpenFace got 93.80% accuracy scores on [LFW data set](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/) whereas human beings could have just 97.53%.

**Similarity**

Face recognition models are regular [convolutional neural networks](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/) and they are responsible to represent faces as vectors. We expect that a face pair of same person should be [more similar](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/) than a face pair of different persons.

Similarity could be calculated by different metrics such as [Cosine Similarity](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration uses cosine similarity.

```python
metrics = ["cosine", "euclidean", "euclidean_l2"]
result = DeepFace.verify("img1.jpg", "img2.jpg", distance_metric = metrics[1])
df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db", distance_metric = metrics[1])
```

Euclidean L2 form [seems](https://youtu.be/i_MOwvhbLdI) to be more stable than cosine and regular Euclidean distance based on experiments.

**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)

Deepface also comes with a strong facial attribute analysis module including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions.

```python
obj = DeepFace.analyze(img_path = "img4.jpg", actions = ['age', 'gender', 'race', 'emotion'])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

Age model got ± 4.65 MAE; gender model got 97.44% accuracy, 96.29% precision and 95.05% recall as mentioned in its [tutorial](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/).

**Streaming and Real Time Analysis** - [`Demo`](https://youtu.be/-c9sSJcx6wI)

You can run deepface for real time videos as well. Stream function will access your webcam and apply both face recognition and facial attribute analysis. The function starts to analyze a frame if it can focus a face sequantially 5 frames. Then, it shows results 5 seconds.

```python
DeepFace.stream(db_path = "C:/User/Sefik/Desktop/database")
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

**Face Detectors** - [`Demo`](https://youtu.be/GZ2p2hj2H5k)

Face detection and alignment are early stages of a modern face recognition pipeline. Experiments show that just alignment increases the face recognition accuracy almost 1%. [`OpenCV`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [`SSD`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/),  [`MTCNN`](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) and [`RetinaFace`](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/) detectors are wrapped in deepface. OpenCV is the default detector.

```python
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

#face detection and alignment
detected_face = DeepFace.detectFace("img.jpg", detector_backend = backends[4])

#face verification
obj = DeepFace.verify("img1.jpg", "img2.jpg", detector_backend = backends[4])

#face recognition
df = DeepFace.find(img_path = "img.jpg", db_path = "my_db", detector_backend = backends[4])

#facial analysis
demography = DeepFace.analyze("img4.jpg", detector_backend = backends[4])
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-detectors.png" width="90%" height="90%"></p>

[RetinaFace](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/) and [MTCNN](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) seem to overperform in detection and alignment stages but they are slower than others. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, if you consider the accuracy, then you should use retinaface or mtcnn.

**Ensemble learning for face recognition** - [`Demo`](https://youtu.be/EIBJJJ0ECXU)

A face recognition task can be handled by several models and similarity metrics. Herein, deepface offers a [special boosting and combination solution](https://sefiks.com/2020/06/03/mastering-face-recognition-with-ensemble-learning/) to improve the accuracy of a face recognition task. This provides a huge improvement on accuracy metrics. On the other hand, this runs much slower than single models.

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

**Tech Stack** - [`Vlog`](https://youtu.be/R8fHsL7u3eE), [`Tutorial`](https://sefiks.com/2021/03/31/tech-stack-recommendations-for-face-recognition/)

Face recognition models represent facial images as vector embeddings. The idea behind facial recognition is that vectors should be more similar for same person than different persons. The question is that where and how to store facial embeddings in a large scale system. Herein, deepface offers a represention function to find vector embeddings from facial images.

```python
embedding = DeepFace.represent("img.jpg", model_name = 'Facenet')
```

Tech stack is vast to store vector embeddings. To determine the right tool, you should consider your task such as face verification or face recognition, priority such as speed or confidence, and also data size.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/tech-stack.png" width="90%" height="90%"></p>

## Contribution

Pull requests are welcome. You should run the unit tests locally by running [`test/unit_tests.py`](https://github.com/serengil/deepface/blob/master/tests/unit_tests.py). Please share the unit test result logs in the PR. Deepface is currently compatible with TF 1 and 2 versions. Change requests should satisfy those requirements both.

## Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one.

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

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details. However, the library wraps some external face recognition models: [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/), [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), [OpenFace](https://github.com/cmusatyalab/openface/blob/master/LICENSE), [DeepFace](https://github.com/swghosh/DeepFace), [DeepID](https://github.com/Ruoyiran/DeepID/blob/master/LICENSE.md), [ArcFace](https://github.com/leondgarse/Keras_insightface/blob/master/LICENSE) and [Dlib](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt). Besides, age, gender and race / ethnicity models are based on VGG-Face. Licence types will be inherited if you are going to use those models.

Deepface [logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/) and it is licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
