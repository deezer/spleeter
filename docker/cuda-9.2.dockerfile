ARG BASE=python:3.7
FROM ${BASE}

# FROM 9.2-base-ubuntu18.04
# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/base/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_VERSION 9.2.148
ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
    && ln -s cuda-9.2 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"

# FROM 9.2-runtime-ubuntu18.04
# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/runtime/Dockerfile
ENV NCCL_VERSION 2.3.7
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda9.2 \
    && apt-mark hold libnccl2 \
    && rm -rf /var/lib/apt/lists/*

# FROM 9.2-runtime-cudnn7-ubuntu18.04
# https://gitlab.com/nvidia/container-images/cuda/blob/ubuntu18.04/9.2/runtime/cudnn7/Dockerfile
ENV CUDNN_VERSION 7.5.0.56
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
    && apt-mark hold libcudnn7 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /model
ENV MODEL_PATH /model
COPY audio_example.mp3 .

# Spleeter installation.
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
RUN pip install musdb museval
RUN pip install spleeter-gpu==1.4.9

ENTRYPOINT ["spleeter"]