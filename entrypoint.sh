#!/bin/bash
set -e

# Parse arguments
MODE="$1"
shift || true # remove first arg (mode), keep the rest
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Default values
PORT=${PORT} # PORT has no default value
WORKERS=${WORKERS:-1} # Workers the number of gunicorn workers or gRPC maximum threads
TIMEOUT=${TIMEOUT:-7200} # Timeout the gunicorn timout (not used by gRPC)
LOG_LEVEL=${LOG_LEVEL:-info} # Log level for the whole application

# Launch requested app
echo "Launching app using parameters mode: $MODE, port: $PORT, workers: $WORKERS, timeout: $TIMEOUT, log level: $LOG_LEVEL."
if [ "$MODE" = "http" ]; then
    gunicorn --workers=$WORKERS --timeout=$TIMEOUT --bind=0.0.0.0:$PORT --log-level=$LOG_LEVEL --access-logformat='%(h)s - - [%(t)s] "%(r)s" %(s)s %(b)s %(L)s' --access-logfile=- "app:create_app()"
elif [ "$MODE" = "grpc" ]; then
    exec python grpc_server.py $@
else
    echo "Unknown mode: $MODE"
    exit 1
fi