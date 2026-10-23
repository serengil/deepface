# usage:
#   ./dockerize.sh                            -> build deepface:latest locally from Dockerfile (onnx backend)
#   ./dockerize.sh source=local               -> same as above, local is the default source
#   ./dockerize.sh source=hub                 -> pull serengil/deepface:latest from docker hub
#   ./dockerize.sh backend=tf                 -> build deepface:tensorflow locally
#   ./dockerize.sh source=hub backend=tf      -> pull serengil/deepface:tensorflow from docker hub
#   ./dockerize.sh source=hub backend=pytorch -> pull serengil/deepface:pytorch from docker hub

# Dockerfile is in the root
cd ..

# start docker
# sudo service docker start

# list current docker packages
# docker container ls -a

# delete existing deepface packages
# docker rm -f $(docker ps -a -q --filter "ancestor=deepface")

# source of the image: local (default) builds it from Dockerfile, hub pulls the pre-built one from docker hub
# backend engine of the image: onnx (default), tf or pytorch
SOURCE="local"
BACKEND="onnx"
for arg in "$@"; do
    case "$arg" in
        source=*) SOURCE="${arg#source=}" ;;
        backend=*) BACKEND="${arg#backend=}" ;;
        *) echo "unknown argument: $arg (expected source=local|hub, backend=onnx|tf|pytorch)"; exit 1 ;;
    esac
done

# each backend has its own image tag: deepface:latest (onnx), deepface:tensorflow or deepface:pytorch
case "$BACKEND" in
    onnx) TAG="latest" ;;
    tf|tensorflow) BACKEND="tensorflow"; TAG="tensorflow" ;;
    pytorch|torch) BACKEND="pytorch"; TAG="pytorch" ;;
    *) echo "unsupported backend: $BACKEND (expected onnx, tf or pytorch)"; exit 1 ;;
esac

case "$SOURCE" in
    local)
        IMAGE="deepface:$TAG"
        echo "building $IMAGE image with $BACKEND backend"
        # build deepface image
        docker build -t "$IMAGE" --build-arg BACKEND="$BACKEND" . || exit 1
        ;;
    hub)
        IMAGE="serengil/deepface:$TAG"
        echo "pulling $IMAGE image with $BACKEND backend from docker hub"
        # pull the pre-built image from docker hub
        docker pull "$IMAGE" || exit 1
        ;;
    *) echo "unsupported source: $SOURCE (expected local or hub)"; exit 1 ;;
esac

# push to docker hub
# docker login
# docker tag "$IMAGE" serengil/"$IMAGE"
# docker push serengil/"$IMAGE"

# e.g. docker tag deepface:latest serengil/deepface:latest && docker push serengil/deepface:latest
# e.g. docker tag deepface:tensorflow serengil/deepface:tensorflow && docker push serengil/deepface:tensorflow

# copy weights from your local
# docker cp ~/.deepface/weights/. <CONTAINER_ID>:/root/.deepface/weights/

# run the built or pulled image
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

# to access the inside of docker image when it is in running status
# docker exec -it <CONTAINER_ID> /bin/sh

# healthcheck
# sleep 3s
# curl localhost:5000
