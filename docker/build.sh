#!/bin/bash

NUM_THREADS=4

orange=`tput setaf 3`
reset_color=`tput sgr0`

echo "Building for ${orange}${reset_color}"

docker build .. \
    -f Dockerfile \
    --build-arg UID=$(id -g) \
    --build-arg GID=$(id -g) \
    --build-arg NUM_THREADS=${NUM_THREADS} \
    -t ms/deit:latest
