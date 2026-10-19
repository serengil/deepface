#!/usr/bin/env bash
cd ../deepface/api/src

# run the service with flask - not for production purposes
# python api.py

# run the service with gunicorn - for prod purposes
if ! command -v gunicorn > /dev/null 2>&1; then
  echo "gunicorn is required to run the deepface api. Please run \`pip install \"deepface[api]\"\` to install it."
  exit 1
fi
gunicorn --workers=1 --timeout=3600 --bind=0.0.0.0:5005 "app:create_app()"