#!/bin/bash

# if [ $# != 1 ]
# then
# 	echo "Usage ./start.sh DATA_DIR_PATH"
# 	echo "Where DATA_DIR_PATH - path to dir that will be mounted to container"
# 	exit 1
# fi

# DATA=$1
CODE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../../ms-deit"

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

echo "Running on ${orange}${ARCH}${reset_color}"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [ "$ARCH" == "aarch64" ] 
then
    ARGS="--runtime nvidia"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

docker run -it -d --rm \
        $ARGS \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name ms-deit \
        --net "host" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $CODE:/home/mindspore/ms-deit:rw \
	# -v $DATA:/data:rw \
    ms/deit
