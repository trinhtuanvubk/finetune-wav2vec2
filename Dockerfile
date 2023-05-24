FROM anibali/pytorch:1.8.1-cuda11.1
USER root

WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Europe/London
ENV HOME=/config
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -q \
  && apt install -y -qq tzdata bash build-essential git curl wget software-properties-common \
    vim ca-certificates libffi-dev libssl-dev libsndfile1 libbz2-dev liblzma-dev locales \
    libboost-all-dev libboost-tools-dev libboost-thread-dev cmake \
    python3 python3-setuptools python3-pip cython \
  && git clone --recursive https://github.com/parlance/ctcdecode.git \
  && cd ctcdecode && pip install . \
  && pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir