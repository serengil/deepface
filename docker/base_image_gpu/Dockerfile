FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update; \
    apt-get install -qy libsm6 libxext6 libxrender-dev; \
    pip install --upgrade pip; \
    pip install --no-cache-dir cmake; \
    pip install --no-cache-dir opencv-python; \
    pip install --no-cache-dir dlib==19.19.0; \
    pip install --no-cache-dir pandas==1.0.0; \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Australia/Melbourne
