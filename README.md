# deepface

<div align="center">

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/deepface?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads)](https://pepy.tech/project/deepface)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/deepface?color=green&label=conda%20downloads)](https://anaconda.org/conda-forge/deepface)
[![Stars](https://img.shields.io/github/stars/serengil/deepface?color=yellow&style=flat)](https://github.com/serengil/deepface/stargazers)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/deepface/blob/master/LICENSE)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dserengil%26type%3Dpatrons&style=flat)](https://www.patreon.com/serengil?repo=deepface)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/serengil?logo=GitHub&color=lightgray)](https://github.com/sponsors/serengil)

[![DOI](http://img.shields.io/:DOI-10.1109/ASYU50717.2020.9259802-blue.svg?style=flat)](https://doi.org/10.1109/ASYU50717.2020.9259802)
[![DOI](http://img.shields.io/:DOI-10.1109/ICEET53442.2021.9659697-blue.svg?style=flat)](https://doi.org/10.1109/ICEET53442.2021.9659697)

[![Blog](https://img.shields.io/:blog-sefiks.com-blue.svg?style=flat&logo=wordpress)](https://sefiks.com)
[![YouTube](https://img.shields.io/:youtube-@sefiks-red.svg?style=flat&logo=youtube)](https://www.youtube.com/@sefiks?sub_confirmation=1)
[![Twitter](https://img.shields.io/:follow-@serengil-blue.svg?style=flat&logo=twitter)](https://twitter.com/serengil)

</div>

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-icon-labeled.png" width="200" height="240"></p>

Deepface is a lightweight [face recognition](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and facial attribute analysis ([age](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [gender](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [emotion](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) and [race](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/)) framework for python. It is a hybrid face recognition framework wrapping **state-of-the-art** models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) and `SFace`.

Experiments show that human beings have 97.53% accuracy on facial recognition tasks whereas those models already reached and passed that accuracy level.

## Installation [![PyPI](https://img.shields.io/pypi/v/deepface.svg)](https://pypi.org/project/deepface/) [![Conda](https://img.shields.io/conda/vn/conda-forge/deepface.svg)](https://anaconda.org/conda-forge/deepface)

The easiest way to install deepface is to download it from [`PyPI`](https://pypi.org/project/deepface/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install deepface
```

Secondly, DeepFace is also available at [`Conda`](https://anaconda.org/conda-forge/deepface). You can alternatively install the package via conda.

```shell
$ conda install -c conda-forge deepface
```

Thirdly, you can install deepface from its source code.

```shell
$ git clone https://github.com/serengil/deepface.git
$ cd deepface
$ pip install -e .
```

Then you will be able to import the library and use its functionalities.

```python
from deepface import DeepFace
```

**Facial Recognition** - [`Demo`](https://youtu.be/WnUVYQP4h44)

A modern [**face recognition pipeline**](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) consists of 5 common stages: [detect](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [align](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [normalize](https://sefiks.com/2020/11/20/facial-landmarks-for-face-recognition-with-dlib/), [represent](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) and [verify](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/). While Deepface handles all these common stages in the background, you don’t need to acquire in-depth knowledge about all the processes behind it. You can just call its verification, find or analysis function with a single line of code.

**Face Verification** - [`Demo`](https://youtu.be/KRCvkNCOphE)

This function verifies face pairs as same person or different persons. It expects exact image paths as inputs. Passing numpy or base64 encoded images is also welcome. Then, it is going to return a dictionary and you should check just its verified key.

```python
result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-1.jpg" width="95%" height="95%"></p>

Verification function can also handle many faces in the face pairs. In this case, the most similar faces will be compared.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/verify-many-faces.jpg" width="95%" height="95%"></p>

**Face recognition** - [`Demo`](https://youtu.be/Hrjp-EStM_s)

[Face recognition](https://sefiks.com/2020/05/25/large-scale-face-recognition-for-deep-learning/) requires applying face verification many times. Herein, deepface has an out-of-the-box find function to handle this action. It's going to look for the identity of input image in the database path and it will return list of pandas data frame as output. Meanwhile, facial embeddings of the facial database are stored in a pickle file to be searched faster in next time. Result is going to be the size of faces appearing in the source image. Besides, target images in the database can have many faces as well.


```python
dfs = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-6-v2.jpg" width="95%" height="95%"></p>

**Embeddings**

Face recognition models basically represent facial images as multi-dimensional vectors. Sometimes, you need those embedding vectors directly. DeepFace comes with a dedicated representation function. Represent function returns a list of embeddings. Result is going to be the size of faces appearing in the image path.

```python
embedding_objs = DeepFace.represent(img_path = "img.jpg")
```

This function returns an array as embedding. The size of the embedding array would be different based on the model name. For instance, VGG-Face is the default model and it represents facial images as 2622 dimensional vectors.

```python
embedding = embedding_objs[0]["embedding"]
assert isinstance(embedding, list)
assert model_name = "VGG-Face" and len(embedding) == 2622
```

Here, embedding is also [plotted](https://sefiks.com/2020/05/01/a-gentle-introduction-to-face-recognition-in-deep-learning/) with 2622 slots horizontally. Each slot is corresponding to a dimension value in the embedding vector and dimension value is explained in the colorbar on the right. Similar to 2D barcodes, vertical dimension stores no information in the illustration.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/embedding.jpg" width="95%" height="95%"></p>

**Face recognition models** - [`Demo`](https://youtu.be/i_MOwvhbLdI)

Deepface is a **hybrid** face recognition package. It currently wraps many **state-of-the-art** face recognition models: [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) , [`Google FaceNet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), [`OpenFace`](https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/), [`Facebook DeepFace`](https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/), [`DeepID`](https://sefiks.com/2020/06/16/face-recognition-with-deepid-in-keras/), [`ArcFace`](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/) and `SFace`. The default configuration uses VGG-Face model.

```python
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

#face verification
result = DeepFace.verify(img1_path = "img1.jpg", 
      img2_path = "img2.jpg", 
      model_name = models[0]
)

#face recognition
dfs = DeepFace.find(img_path = "img1.jpg",
      db_path = "C:/workspace/my_db", 
      model_name = models[1]
)

#embeddings
embedding_objs = DeepFace.represent(img_path = "img.jpg", 
      model_name = models[2]
)
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/model-portfolio-v8.jpg" width="95%" height="95%"></p>

FaceNet, VGG-Face, ArcFace and Dlib are [overperforming](https://youtu.be/i_MOwvhbLdI) ones based on experiments. You can find out the scores of those models below on both [Labeled Faces in the Wild](https://sefiks.com/2020/08/27/labeled-faces-in-the-wild-for-face-recognition/) and YouTube Faces in the Wild data sets declared by its creators.

| Model | LFW Score | YTF Score |
| ---   | --- | --- |
| Facenet512 | 99.65% | - |
| SFace | 99.60% | - |
| ArcFace | 99.41% | - |
| Dlib | 99.38 % | - |
| Facenet | 99.20% | - |
| VGG-Face | 98.78% | 97.40% |
| *Human-beings* | *97.53%* | - |
| OpenFace | 93.80% | - |
| DeepID | - | 97.05% |

**Similarity**

Face recognition models are regular [convolutional neural networks](https://sefiks.com/2018/03/23/convolutional-autoencoder-clustering-images-with-neural-networks/) and they are responsible to represent faces as vectors. We expect that a face pair of same person should be [more similar](https://sefiks.com/2020/05/22/fine-tuning-the-threshold-in-face-recognition/) than a face pair of different persons.

Similarity could be calculated by different metrics such as [Cosine Similarity](https://sefiks.com/2018/08/13/cosine-similarity-in-machine-learning/), Euclidean Distance and L2 form. The default configuration uses cosine similarity.

```python
metrics = ["cosine", "euclidean", "euclidean_l2"]

#face verification
result = DeepFace.verify(img1_path = "img1.jpg", 
          img2_path = "img2.jpg", 
          distance_metric = metrics[1]
)

#face recognition
dfs = DeepFace.find(img_path = "img1.jpg", 
          db_path = "C:/workspace/my_db", 
          distance_metric = metrics[2]
)
```

Euclidean L2 form [seems](https://youtu.be/i_MOwvhbLdI) to be more stable than cosine and regular Euclidean distance based on experiments.

**Facial Attribute Analysis** - [`Demo`](https://youtu.be/GT2UeN85BdA)

Deepface also comes with a strong facial attribute analysis module including [`age`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`gender`](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/), [`facial expression`](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/) (including angry, fear, neutral, sad, disgust, happy and surprise) and [`race`](https://sefiks.com/2019/11/11/race-and-ethnicity-prediction-in-keras/) (including asian, white, middle eastern, indian, latino and black) predictions. Result is going to be the size of faces appearing in the source image.

```python
objs = DeepFace.analyze(img_path = "img4.jpg", 
        actions = ['age', 'gender', 'race', 'emotion']
)
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/stock-2.jpg" width="95%" height="95%"></p>

Age model got ± 4.65 MAE; gender model got 97.44% accuracy, 96.29% precision and 95.05% recall as mentioned in its [tutorial](https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/).


**Face Detectors** - [`Demo`](https://youtu.be/GZ2p2hj2H5k)

Face detection and alignment are important early stages of a modern face recognition pipeline. Experiments show that just alignment increases the face recognition accuracy almost 1%. [`OpenCV`](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/), [`SSD`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/), [`Dlib`](https://sefiks.com/2020/07/11/face-recognition-with-dlib-in-python/),  [`MTCNN`](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/), [`RetinaFace`](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/), [`MediaPipe`](https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/), [`YOLOv8 Face`](https://github.com/derronqi/yolov8-face) and [`YuNet`](https://github.com/ShiqiYu/libfacedetection) detectors are wrapped in deepface.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/detector-portfolio-v5.jpg" width="95%" height="95%"></p>

All deepface functions accept an optional detector backend input argument. You can switch among those detectors with this argument. OpenCV is the default detector.

```python
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
]

#face verification
obj = DeepFace.verify(img1_path = "img1.jpg", 
        img2_path = "img2.jpg", 
        detector_backend = backends[0]
)

#face recognition
dfs = DeepFace.find(img_path = "img.jpg", 
        db_path = "my_db", 
        detector_backend = backends[1]
)

#embeddings
embedding_objs = DeepFace.represent(img_path = "img.jpg", 
        detector_backend = backends[2]
)

#facial analysis
demographies = DeepFace.analyze(img_path = "img4.jpg", 
        detector_backend = backends[3]
)

#face detection and alignment
face_objs = DeepFace.extract_faces(img_path = "img.jpg", 
        target_size = (224, 224), 
        detector_backend = backends[4]
)
```

Face recognition models are actually CNN models and they expect standard sized inputs. So, resizing is required before representation. To avoid deformation, deepface adds black padding pixels according to the target size argument after detection and alignment.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-detectors-v3.jpg" width="90%" height="90%"></p>

[RetinaFace](https://sefiks.com/2021/04/27/deep-face-detection-with-retinaface-in-python/) and [MTCNN](https://sefiks.com/2020/09/09/deep-face-detection-with-mtcnn-in-python/) seem to overperform in detection and alignment stages but they are much slower. If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, if you consider the accuracy, then you should use retinaface or mtcnn.

The performance of RetinaFace is very satisfactory even in the crowd as seen in the following illustration. Besides, it comes with an incredible facial landmark detection performance. Highlighted red points show some facial landmarks such as eyes, nose and mouth. That's why, alignment score of RetinaFace is high as well.

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/retinaface-results.jpeg" width="90%" height="90%">
<br><em>The Yellow Angels - Fenerbahce Women's Volleyball Team</em>
</p>

You can find out more about RetinaFace on this [repo](https://github.com/serengil/retinaface).

**Real Time Analysis** - [`Demo`](https://youtu.be/-c9sSJcx6wI)

You can run deepface for real time videos as well. Stream function will access your webcam and apply both face recognition and facial attribute analysis. The function starts to analyze a frame if it can focus a face sequentially 5 frames. Then, it shows results 5 seconds.

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

**API** - [`Demo`](https://youtu.be/HeKCQ6U9XmI)

DeepFace serves an API as well. You can clone [`/api`](https://github.com/serengil/deepface/tree/master/api) folder and run the api via gunicorn server. This will get a rest service up. In this way, you can call deepface from an external system such as mobile app or web.

```shell
cd scripts
./service.sh
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-api.jpg" width="90%" height="90%"></p>

Face recognition, facial attribute analysis and vector representation functions are covered in the API. You are expected to call these functions as http post methods. Default service endpoints will be `http://localhost:5000/verify` for face recognition, `http://localhost:5000/analyze` for facial attribute analysis, and `http://localhost:5000/represent` for vector representation. You can pass input images as exact image paths on your environment, base64 encoded strings or images on web. [Here](https://github.com/serengil/deepface/tree/master/api), you can find a postman project to find out how these methods should be called.

**Dockerized Service**

You can deploy the deepface api on a kubernetes cluster with docker. The following [shell script](https://github.com/serengil/deepface/blob/master/scripts/dockerize.sh) will serve deepface on `localhost:5000`. You need to re-configure the [Dockerfile](https://github.com/serengil/deepface/blob/master/Dockerfile) if you want to change the port. Then, even if you do not have a development environment, you will be able to consume deepface services such as verify and analyze. You can also access the inside of the docker image to run deepface related commands. Please follow the instructions in the [shell script](https://github.com/serengil/deepface/blob/master/scripts/dockerize.sh).

```shell
cd scripts
./dockerize.sh
```

<p align="center"><img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/deepface-dockerized-v2.jpg" width="50%" height="50%"></p>

**Command Line Interface**

DeepFace comes with a command line interface as well. You are able to access its functions in command line as shown below. The command deepface expects the function name as 1st argument and function arguments thereafter.

```shell
#face verification
$ deepface verify -img1_path tests/dataset/img1.jpg -img2_path tests/dataset/img2.jpg

#facial analysis
$ deepface analyze -img_path tests/dataset/img1.jpg
```

You can also run these commands if you are running deepface with docker. Please follow the instructions in the [shell script](https://github.com/serengil/deepface/blob/master/scripts/dockerize.sh#L17).

## Contribution [![Tests](https://github.com/serengil/deepface/actions/workflows/tests.yml/badge.svg)](https://github.com/serengil/deepface/actions/workflows/tests.yml)

Pull requests are more than welcome! You should run the unit tests locally by running [`test/unit_tests.py`](https://github.com/serengil/deepface/blob/master/tests/unit_tests.py) before creating a PR. Once a PR sent, GitHub test workflow will be run automatically and unit test results will be available in [GitHub actions](https://github.com/serengil/deepface/actions) before approval. Besides, workflow will evaluate the code with pylint as well.

## Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one 🙏

You can also support this work on [Patreon](https://www.patreon.com/serengil?repo=deepface) or [GitHub Sponsors](https://github.com/sponsors/serengil).

<a href="https://www.patreon.com/serengil?repo=deepface">
<img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/patreon.png" width="30%" height="30%">
</a>

## Citation

Please cite deepface in your publications if it helps your research. Here are its BibTex entries:

If you use deepface for facial recogntion purposes, please cite the this publication.

```BibTeX
@inproceedings{serengil2020lightface,
  title        = {LightFace: A Hybrid Deep Face Recognition Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages        = {23-27},
  year         = {2020},
  doi          = {10.1109/ASYU50717.2020.9259802},
  url          = {https://doi.org/10.1109/ASYU50717.2020.9259802},
  organization = {IEEE}
}
```

 If you use deepface for facial attribute analysis purposes such as age, gender, emotion or ethnicity prediction or face detection purposes, please cite the this publication.

```BibTeX
@inproceedings{serengil2021lightface,
  title        = {HyperExtended LightFace: A Facial Attribute Analysis Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2021 International Conference on Engineering and Emerging Technologies (ICEET)},
  pages        = {1-4},
  year         = {2021},
  doi          = {10.1109/ICEET53442.2021.9659697},
  url          = {https://doi.org/10.1109/ICEET53442.2021.9659697},
  organization = {IEEE}
}
```

Also, if you use deepface in your GitHub projects, please add `deepface` in the `requirements.txt`.

## Licence

Deepface is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/deepface/blob/master/LICENSE) for more details. However, the library wraps some external face recognition models: [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/), [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), [OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/LICENSE), [DeepFace](https://github.com/swghosh/DeepFace), [DeepID](https://github.com/Ruoyiran/DeepID/blob/master/LICENSE.md), [ArcFace](https://github.com/leondgarse/Keras_insightface/blob/master/LICENSE), [Dlib](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt), and [SFace](https://github.com/opencv/opencv_zoo/blob/master/models/face_recognition_sface/LICENSE). Besides, age, gender and race / ethnicity models are based on VGG-Face. Licence types will be inherited if you are going to use those models. Please check the license types of those models for production purposes.

Deepface [logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/) and it is licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
