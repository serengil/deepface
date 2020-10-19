#!/usr/bin/env bash

GIT_VERSION=$(git describe --tags --always --long)

docker build -t repo.eresearch.unimelb.edu.au:8001/happypet/webapp:$GIT_VERSION \
             -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp:$GIT_VERSION \
             -t repo.eresearch.unimelb.edu.au:8001/happypet/webapp:latest \
             -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp:latest \
             -f docker/release/Dockerfile .

docker build -t repo.eresearch.unimelb.edu.au:8001/happypet/webapp-gpu:$GIT_VERSION \
             -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp-gpu:$GIT_VERSION \
             -t repo.eresearch.unimelb.edu.au:8001/happypet/webapp-gpu:latest \
             -t repo.eresearch.unimelb.edu.au:8000/happypet/webapp-gpu:latest \
             -f docker/release_gpu/Dockerfile .
