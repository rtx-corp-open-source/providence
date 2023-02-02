# **Raytheon Technologies proprietary**
# Export controlled - see license file

FROM nvidia/cuda:10.2-cudnn8-runtime as cuda-base

LABEL maintaner.name="Tynan Daly 😈" \
      maintaner.email="tynan.dalt@rtx.com" \
      release="0.0.0" \
      build_date=${DATE}

# Proxy vars
ENV http_proxy=${HTTP_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true
ENV PATH=$PATH:/root/miniconda3/bin

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    zip \
    unzip \
    ca-certificates \
    sudo \
    wget \
    git \
    bzip2 \
    libx11-6 \
 && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm awscliv2.zip \
 && rm -rf /var/lib/apt/lists/*

# Install python 3.8
# I don't like using conda much but deadsnakes is throwing me extremely weird errors
from cuda-base as python-base

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  mkdir /root/.conda && \
  bash Miniconda3-latest-Linux-x86_64.sh -b && \
  rm -f Miniconda3-latest-Linux-x86_64.sh && \
  conda install python=3.8.12



FROM python-base as dl-requirements


COPY requirements.txt requirements-dev.txt ./

# Ensure we're running 64 bit python
RUN python -c "import platform; print(platform.architecture())"

RUN pip install -r requirements.txt \
    && pip install -r requirements-dev.txt \
    # the following comments are for cuda11. For some reason pip cannot find 1.7.1+cu110
    # so we needed to downgrade our base image to cuda102
    #torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \
    #&& pip install --upgrade jax jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app
