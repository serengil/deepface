# Dockerfile is in the root
cd ..

# start docker
# sudo service docker start

# list current docker packages
# docker container ls -a

# delete existing deepface packages
# docker rm -f $(docker ps -a -q --filter "ancestor=deepface")

# backend engine of the image: tf (default), pytorch or onnx
# usage: ./dockerize.sh, ./dockerize.sh backend=pytorch or ./dockerize.sh backend=onnx
BACKEND="tf"
for arg in "$@"; do
    case "$arg" in
        backend=*) BACKEND="${arg#backend=}" ;;
        *) echo "unknown argument: $arg (expected backend=tf, backend=pytorch or backend=onnx)"; exit 1 ;;
    esac
done

# each backend has its own image tag: deepface:latest, deepface:pytorch or deepface:onnx
case "$BACKEND" in
    tf|tensorflow) BACKEND="tensorflow"; TAG="latest" ;;
    pytorch|torch) BACKEND="pytorch"; TAG="pytorch" ;;
    onnx) TAG="onnx" ;;
    *) echo "unsupported backend: $BACKEND (expected tf, pytorch or onnx)"; exit 1 ;;
esac
IMAGE="deepface:$TAG"
echo "building $IMAGE image with $BACKEND backend"

# build deepface image
docker build -t "$IMAGE" --build-arg BACKEND="$BACKEND" . || exit 1

# push to docker hub
# docker login
# docker tag "$IMAGE" serengil/"$IMAGE"
# docker push serengil/"$IMAGE"

# e.g. tf is still default -> docker tag deepface:latest serengil/deepface:latest && docker push serengil/deepface:latest
# e.g. docker tag deepface:onnx serengil/deepface:onnx && docker push serengil/deepface:onnx

# copy weights from your local
# docker cp ~/.deepface/weights/. <CONTAINER_ID>:/root/.deepface/weights/

# run the built image
# docker run --net="host" "$IMAGE"
# docker run -p 5005:5000 "$IMAGE"
ENV_FILE="deepface/api/.env"
if [ -f "$ENV_FILE" ]; then
    echo ".env found, sending to container"
    docker run -p 5005:5000 --env-file "$ENV_FILE" "$IMAGE"
else
    echo "no .env found, running container without env vars"
    docker run -p 5005:5000 "$IMAGE"
fi


# or pull the pre-built image from docker hub and run it
# docker pull serengil/deepface (or serengil/deepface:pytorch, serengil/deepface:onnx)
# docker run -p 5005:5000 serengil/deepface

# to access the inside of docker image when it is in running status
# docker exec -it <CONTAINER_ID> /bin/sh

# healthcheck
# sleep 3s
# curl localhost:5000