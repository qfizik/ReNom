FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# update apt info & install packages
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    python3 \
    python3-tk \
    build-essential \
    python3-pip \
    python3-venv \
    nodejs \
    npm \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install pip packages
RUN pip3 install jupyter numpy cython

# install n
RUN npm install -g n

# install node stabled version
RUN n stable

# export env
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH" 
ENV CUDA_HOME="/usr/local/cuda"
ENV RENOM_PRECISION=32

# git clone ReNom Apps
RUN pip3 install -U pip
RUN pip3 install pytest
RUN git clone https://oauth2:unjK94-AP2HFqBSk5p7z@gitlab.com/grid-devs/ReNom.git \
    && cd ReNom && pip3 install -r requirements.txt \
    && python3 setup.py build_ext -if && pip3 install -e . \
    && python3 renom/cuda/__init__.py
