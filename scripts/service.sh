#!/usr/bin/env bash
cd ../api/src
gunicorn --workers=1 --timeout=3600 --bind=0.0.0.0:5000 "app:create_app()"