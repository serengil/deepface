# start docker
# sudo service docker start

# list current docker packages
# docker container ls -a

# delete existing deepface packages
# docker rm -f $(docker ps -a -q --filter "ancestor=deepface")

# build deepface image
docker build -t deepface_image .

# run image
docker run --net="host" deepface

# to access the inside of docker image when it is in running status
# docker run -it --net="host" deepface /bin/sh

# healthcheck
# sleep 3s
# curl localhost:5000