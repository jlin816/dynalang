# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh scripts/run_messenger_s1.sh EXP_NAME GPU_IDS SEED

# System
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb curl pkg-config \
  libhdf5-dev libfreetype6-dev zip \
  libsdl1.2-dev libsdl-image1.2-dev libsdl-ttf2.0-dev \
  libsdl-mixer1.2-dev libportmidi-dev \
  && apt-get clean

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n dynalang python=3.8
# Automatically use the conda env for any RUN commands
SHELL ["conda", "run", "-n", "dynalang", "/bin/bash", "-c"]

RUN pip3 install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# HomeGrid
RUN pip install homegrid

# Messenger: Change `messenger-emma` to your local messenger-emma repo path (must be in the Docker build context).
COPY messenger-emma /messenger-emma
RUN pip install vgdl@git+https://github.com/ahjwang/py-vgdl
RUN cd /messenger-emma; pip install -e .

# Uncomment if running VLN
# COPY dynalang/env_vln.yml /environment.yml
# RUN pip install "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install torch torchvision
# RUN conda env update -f env_vln.yml
# RUN conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

# Google Cloud DNS cache (optional)
# ENV GCS_RESOLVE_REFRESH_SECS=60
# ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
# ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
# ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
# ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600
# ENV GOOGLE_CLOUD_BUCKET_NAME=your_bucket_name
# ENV WANDB_API_KEY=your_wandb_key

COPY dynalang /dynalang
WORKDIR dynalang
RUN chown -R 1000:root /dynalang && chmod -R 775 /dynalang
RUN pip install -e /dynalang
