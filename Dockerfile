# base image
FROM python:3.8.12
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface



# -----------------------------------
# switch to application directory
WORKDIR /app

# -----------------------------------
# update image os
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
# even though we will use local requirements, this one is required to perform install deepface from source code
COPY ./requirements.txt /app/requirements.txt
# setup.py reads requirements of the backend engine extras as well
COPY ./requirements_tf.txt /app/requirements_tf.txt
COPY ./requirements_pytorch.txt /app/requirements_pytorch.txt
COPY ./requirements_onnx.txt /app/requirements_onnx.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

# -----------------------------------
# if you plan to use a GPU, you should install the 'tensorflow-gpu' package
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org tensorflow-gpu

# if you plan to use face anti-spoofing, then activate this line
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org torch==2.1.2
# -----------------------------------
# install deepface from pypi release (might be out-of-date)
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org deepface
# -----------------------------------
# install dependencies - deepface with these dependency versions is working
# backend engine is chosen with a build arg: tensorflow (default), pytorch or onnx
#   docker build -t deepface .                               -> tensorflow
#   docker build -t deepface --build-arg BACKEND=pytorch .   -> pytorch
#   docker build -t deepface --build-arg BACKEND=onnx .      -> onnx
# for pytorch and onnx, tensorflow and its dependent packages are not installed, so
# retinaface and mtcnn detectors are not available - opencv is the default one anyway.
# onnx builds the smallest image as it runs on onnxruntime only.
ARG BACKEND=tensorflow
RUN if [ "$BACKEND" = "tensorflow" ]; then \
        pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt; \
    elif [ "$BACKEND" = "pytorch" ]; then \
        grep -vE "^(tensorflow|keras|mtcnn|retina-face)==" /app/requirements_local.txt > /app/requirements_local_notf.txt && \
        pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local_notf.txt torch==2.1.2; \
    elif [ "$BACKEND" = "onnx" ]; then \
        grep -vE "^(tensorflow|keras|mtcnn|retina-face)==" /app/requirements_local.txt > /app/requirements_local_notf.txt && \
        pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local_notf.txt onnxruntime==1.16.3; \
    else \
        echo "Unsupported BACKEND=$BACKEND. It must be tensorflow, pytorch or onnx." && exit 1; \
    fi
ENV DEEPFACE_BACKEND_ENGINE=$BACKEND

# install deepface from source code (always up-to-date)
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e . --no-deps

# -----------------------------------
# some packages are optional in deepface. activate if your task depends on one.
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org cmake==3.24.1.1
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dlib==19.20.0
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org lightgbm==2.3.1

# -----------------------------------
# if you plan to serve deepface over grpc, then activate these lines to generate grpc stubs
# also activate grpc dependencies from requirements_local
# COPY ./Makefile /app/Makefile
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org grpcio==1.62.3 grpcio-tools==1.62.3
# RUN make grpc

# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1

# -----------------------------------
# run the app (re-configure port if necessary)
WORKDIR /app/deepface/api/src
EXPOSE 5000
# activate this line if you plan to serve deepface over grpc
# EXPOSE 50051
# CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
ENTRYPOINT [ "sh", "entrypoint.sh" ]
