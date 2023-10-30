# start docker
# sudo service docker start

# list current docker packages
# docker container ls -a

# delete existing deepface packages
# docker rm -f $(docker ps -a -q --filter "ancestor=deepface")

# build deepface image
docker build -t deepface .

# copy weights from your local
# docker cp ~/.deepface/weights/. <CONTAINER_ID>:/root/.deepface/weights/

# run image
docker run --name deepface -itd -p 5000:5000 deepface

# to access the inside of docker image when it is in running status
# docker exec -it <CONTAINER_ID> /bin/sh

# healthcheck
# sleep 3s
# curl localhost:5000