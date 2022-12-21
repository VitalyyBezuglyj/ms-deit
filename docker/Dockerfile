FROM swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:1.9.0

ENV DEBIAN_FRONTEND noninteractive

RUN set -x && \
  apt-get update -y && \
  apt-get upgrade -y --no-install-recommends && \
  : "basic dependencies" && \
  apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    tar \
    unzip \
    vim

# Expose default SSH port
EXPOSE 22

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get install -y \
    sudo \
    tmux \
    openssh-server

# User setup
ARG USER=mindspore
ARG UID=1000
ARG GID=1000
ARG PW=user

# Add user and his password
RUN useradd -m ${USER} --uid=${UID} \
    && echo "${USER}:${PW}" | chpasswd \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER}

ENV WORKDIR /home/${USER}
WORKDIR /home/${USER}
COPY . ${WORKDIR}/ms-deit
RUN pwd && pip install -r ms-deit/requriments.txt