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


# # Add a docker user so we that created files in the docker container are owned by a non-root user
# RUN addgroup --gid 1000 docker && \
#     adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
#     echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# # Remap the docker user and group to be the same uid and group as the host user.
# # Any created files by the docker container will be owned by the host user.
# RUN USER=docker && \
#     GROUP=docker && \
#     curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
#     chown root:root /usr/local/bin/fixuid && \
#     chmod 4755 /usr/local/bin/fixuid && \
#     mkdir -p /etc/fixuid && \
#     printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

# USER docker:docker


ADD OpenPCDet /home/OpenPCDet
# ADD spconv /home/spconv

RUN pip3 install --upgrade pip && \
    pip3 install numba==0.50.0 scikit-image scipy pillow==8.0 && \
    pip3 install fire tensorboardX protobuf opencv-python torch==1.5 torchvision nuscenes-devkit && \
    pip3 install easydict pyyaml==5.4.1 spconv-cu102	

WORKDIR "/home/OpenPCDet"
RUN python3 setup.py develop 

CMD ["bash"]