# Adapted from https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# Updated to use CUDA 11.3 with Ubuntu 20.04 (10.1 image no longer available)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build python3-pip
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN python3 -m pip install --user --upgrade pip

# git clone DASO
RUN git clone https://github.com/ytaek-oh/daso.git && cd daso

# install dependencies
# Updated PyTorch to 1.12.1 which is compatible with CUDA 11.3
RUN pip install --user tensorboard
RUN pip install --user torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install --user yacs
