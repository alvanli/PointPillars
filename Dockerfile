# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

USER root:root

WORKDIR /home/

SHELL ["/bin/bash", "-c"]

ENV PATH="/home/miniconda3/bin:${PATH}"
ARG PATH="/home/miniconda3/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN apt-get install -y --force-yes software-properties-common
RUN apt-get install -y git wget curl
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install -y build-essential libpq-dev libssl-dev OpenSSL libffi-dev zlib1g-dev
RUN apt-get install -y python3-pip

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b -p /home/miniconda3 

ADD second.pytorch /home/PointPillars

RUN pip install numba scikit-image scipy pillow && \
    pip install fire tensorboardX protobuf opencv-python torch torchvision nuscenes-devkit && \
    pip install spconv-cu111

# RUN cd /home/PointPillars && conda install scikit-image scipy numba pillow matplotlib && \
#     pip install fire tensorboardX protobuf opencv-python nuscenes-devkit && \
#     conda install pytorch torchvision -c pytorch && \
#     pip install spconv-cu111

# cd PointPillars/second.pytorch/second
# RUN conda activate pointpillars && python3 create_data.py nuscenes_data_prep --root_path=/home/nuScenes/v1.0-trainval --dataset_name="nuscenes" --version="v1.0-trainval" --max_sweeps=10

# RUN apt-get install -y libboost-all-dev

CMD ["bash"]