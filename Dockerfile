FROM python:3.8

LABEL org.opencontainers.image.source https://github.com/serengil/deepface

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install .

CMD ["deepface", "--help"]