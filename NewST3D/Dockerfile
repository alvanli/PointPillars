# syntax=docker/dockerfile:1
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

USER root:root

WORKDIR /home/

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN apt-get purge --auto-remove python3 && apt-get install -y python3.8
RUN apt-get install -y --force-yes software-properties-common lsb-release
RUN apt-get install -y git wget curl
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y python3-pip
RUN apt-get install -y libboost-all-dev
RUN apt-get update && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update && apt-get install kitware-archive-keyring && rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-get update && apt-get -y install cmake protobuf-compiler 

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -p /home/miniconda3 

ADD OpenPCDet /home/OpenPCDet
# ADD spconv /home/spconv

RUN pip3 install --upgrade pip && \
    pip3 install numba==0.50.0 scikit-image scipy pillow==8.0 && \
    pip3 install fire tensorboardX protobuf opencv-python torch==1.5 torchvision nuscenes-devkit && \
    pip3 install easydict pyyaml==5.4.1 spconv-cu102	
#spconv-cu102 

# WORKDIR "/home/spconv"
# RUN python3 setup.py bdist_wheel 

# WORKDIR "/home/spconv/dist"
# RUN pip3 install *

WORKDIR "/home/OpenPCDet"
RUN python3 setup.py develop 

CMD ["bash"]