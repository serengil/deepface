#!/usr/bin/env bash

DOCKER_BUILDKIT=1 docker build -t repo.eresearch.unimelb.edu.au:8001/happypet/tensorflow:1.15.2-py3 \
                               -t repo.eresearch.unimelb.edu.au:8000/happypet/tensorflow:1.15.2-py3 \
                               -f docker/base_image/Dockerfile .

DOCKER_BUILDKIT=1 docker build -t repo.eresearch.unimelb.edu.au:8001/happypet/tensorflow:1.15.2-gpu-py3 \
                               -t repo.eresearch.unimelb.edu.au:8000/happypet/tensorflow:1.15.2-gpu-py3 \
                               -f docker/base_image_gpu/Dockerfile .
