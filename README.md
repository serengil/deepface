# WebSite

## About

Under the FlaksServer directory, there are 9 folders. The Breed_Cat, Breed_Dog, Cat_vs_Dog, Emotion_Cat and Emotion_Dog folder  contain the corresponding models and helper functions. The scripts directory contaisn the files which using the above models to predict. The static folder contains the images resources and the css and js libraries. The templates directory contains the html files. The testImages file used for temporarily store the image for prediction. The app.py file contains the server functions. Get trained models from [models](https://drive.google.com/drive/folders/19c2oPX0XAdVnRjaE3_o9EvLeQ4EyRzII?usp=sharing)

### Features

    Python Flask,
    TensorFlow

### Dependencies

    Python 3.7.7,
    TensorFlow 1.15.0

### Env variables

    HAP_DB_NAME=hap
    HAP_DB_HOST=localhost
    HAP_DB_PORT=27017
    HAP_DB_USERNAME
    HAP_DB_PASSWORD


## Deployment

### Run on local

1. git clone the repo
2. Enable the python env 3.7.7
3. All dependencies are in requirements.txt. Run `pip install -r requirements.txt` to install all project dependencies (If any error comes from cmake dlib then run pip3 install cmake).
4. To run the app: `python src/app.py`
5. Hit the endpoint `http://localhost:5000/`

### Run with Flask

For running the back-end server with Flask:

`pip install -r requirements.txt`

`export FLASK_APP=app.py`

`flask run --host=0.0.0.0`

Then you can visit the website on your localhost and the port is 5000. But run on this way, only one user can access the website simultaneously.

### Run with Gunicorn

Using Gunicorn to run the server:

`pip install -r requirements.txt`

`gunicorn -b 0.0.0.0:5000 -t 60 app:app`

Then you can visit the website on your localhost and the port is 5000. On this way, the website can be visited by multiple users simultaneously.

### Run with Docker

#### Build the base image

Use the following command to build the base image (Docker version 19.03 or above):

Without GPU support:

`DOCKER_BUILDKIT=1 docker build -t repo.eresearch.unimelb.edu.au:8000/happypet/tensorflow:1.15.2-py3 -f docker/base_image/Dockerfile .`

With GPU support:

`DOCKER_BUILDKIT=1 docker build -t repo.eresearch.unimelb.edu.au:8000/happypet/tensorflow:1.15.2-gpu-py3 -f docker/base_image/Dockerfile .`

#### Build the release image

Use the following command to build the release image:

Without GPU support:

`docker build -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp:latest -f docker/release/Dockerfile .`

With GPU support:

`docker build -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp:latest -f docker/release/Dockerfile .`

#### Run Docker container

Without GPU support:

`docker run -p 5000:5000 repo.eresearch.unimelb.edu.au:8000/happypet/webapp:latest`

With GPU support:

`docker run -p 5000:5000 repo.eresearch.unimelb.edu.au:8000/happypet/webapp-gpu:latest`
